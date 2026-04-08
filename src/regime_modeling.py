"""Phase 6c: Regime Detection + Conditional Modeling.

Motivation: EDA (Phase 2.3) shows responder_6 std varies from 0.81 to 1.01
across time periods, indicating regime shifts. The lag relationship may also
differ by regime — in high-volatility periods, the autocorrelation might be
stronger or weaker.

This script:
1. Builds a regime classifier based on recent market state (volatility, trend)
2. Trains separate GRU heads per regime (shared encoder, regime-specific output)
3. Tests whether regime-aware modeling improves validation R²

Regime features (computed from recent history, NO future leakage):
  - Rolling volatility (std of responder_6 lag, window=50 steps)
  - Rolling mean of responder_6 lag (trend indicator)
  - Cross-sectional dispersion of features (market stress indicator)
  - Time-of-day effects (time_id bins)

Usage:
    python regime_modeling.py              # full regime analysis
"""

import gc
import json
import time
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import weighted_r2
from neural_network import (
    TRAIN_PATH, ALL_FEATURES, LAG_R6_IDX, LAG_FEAT_COLS,
    SELECTED_54, LAG_RESPONDERS, TARGET_COL,
    TRAIN_LO, TRAIN_HI, VAL_LO, VAL_HI, VAL_OVERLAP_START,
    DEVICE, GRUModel,
    load_pass, standardize, predict_gru,
    find_sequence_groups, create_chunks, ChunkedSeqDataset, collate_sequences,
    BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, PATIENCE,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "regime_modeling"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_REGIMES = 3  # Low-vol, Medium-vol, High-vol
results = {}


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)


# ─── Regime Feature Engineering ─────────────────────────────────────────────

def compute_regime_features(X, date_ids, time_ids, symbol_ids, feat_mean, feat_std):
    """Compute regime indicator features from the data.

    These features are computed from CURRENT and RECENT history only — no future leakage.

    Returns: regime_labels (N,) array with values 0, 1, 2
             regime_features (N, n_regime_features) array
    """
    n = len(X)
    lag_r6_idx = ALL_FEATURES.index("lag1_r6")

    # Un-normalize lag1_r6 to get approximate responder_6 level
    lag_r6 = X[:, lag_r6_idx] * feat_std[lag_r6_idx] + feat_mean[lag_r6_idx]

    # Regime features per (symbol, date) group:
    # 1. Rolling volatility (std of lag_r6 over last 50 steps within symbol)
    # 2. Rolling mean (trend of lag_r6 over last 50 steps)
    # 3. Cross-sectional feature dispersion at each time step

    groups = find_sequence_groups(symbol_ids, date_ids)

    # Per-symbol rolling features
    rolling_vol = np.zeros(n, dtype=np.float32)
    rolling_mean = np.zeros(n, dtype=np.float32)

    for g_start, g_end in groups:
        g_len = g_end - g_start
        lag_vals = lag_r6[g_start:g_end]

        for i in range(g_len):
            window_start = max(0, i - 50)
            window = lag_vals[window_start:i + 1]
            rolling_vol[g_start + i] = window.std() if len(window) > 2 else 0.0
            rolling_mean[g_start + i] = window.mean()

    # Cross-sectional feature dispersion per time step
    # Use top 6 features (highest LightGBM gain)
    top_feat_indices = [ALL_FEATURES.index(f"feature_{f}") for f in
                        ["06", "61", "30", "36", "07", "04"]]

    # Group by (date_id, time_id) and compute std across symbols
    time_keys = date_ids.astype(np.int64) * 10000 + time_ids
    unique_times = np.unique(time_keys)

    cs_dispersion = np.zeros(n, dtype=np.float32)
    cs_mean_feat = np.zeros(n, dtype=np.float32)

    for tk in unique_times:
        mask = time_keys == tk
        if mask.sum() < 2:
            continue
        # Feature dispersion: mean std of top features across symbols
        X_subset = X[np.where(mask)[0]][:, top_feat_indices]
        cs_dispersion[mask] = X_subset.std(axis=0).mean()
        cs_mean_feat[mask] = X_subset.mean()

    # Regime features array
    regime_features = np.column_stack([
        rolling_vol,       # 0: symbol-level volatility
        rolling_mean,      # 1: symbol-level trend
        cs_dispersion,     # 2: cross-sectional feature dispersion
        cs_mean_feat,      # 3: cross-sectional feature mean
        (time_ids % 921).astype(np.float32) / 921.0,  # 4: normalized time-of-day (921 avg slots/day)
    ])

    # Classify regimes based on rolling volatility
    # Use percentiles of rolling_vol (computed per-symbol to be fair)
    vol_by_symbol = {}
    for g_start, g_end in groups:
        sym = symbol_ids[g_start]
        if sym not in vol_by_symbol:
            vol_by_symbol[sym] = []
        vol_by_symbol[sym].extend(rolling_vol[g_start:g_end])

    # Global volatility percentiles
    all_vols = np.concatenate(list(vol_by_symbol.values()))
    p33 = np.percentile(all_vols, 33)
    p66 = np.percentile(all_vols, 66)

    regime_labels = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if rolling_vol[i] < p33:
            regime_labels[i] = 0  # Low vol
        elif rolling_vol[i] < p66:
            regime_labels[i] = 1  # Medium vol
        else:
            regime_labels[i] = 2  # High vol

    regime_names = ["low_vol", "medium_vol", "high_vol"]
    regime_counts = [(regime_labels == i).sum() for i in range(N_REGIMES)]

    print(f"  Regime classification (volatility-based):")
    for i, name in enumerate(regime_names):
        pct = regime_counts[i] / n * 100
        print(f"    {name}: {regime_counts[i]:,} rows ({pct:.1f}%)")
    print(f"  Volatility thresholds: p33={p33:.4f}, p66={p66:.4f}")

    return regime_labels, regime_features, regime_names


# ─── Regime-Conditional GRU Model ──────────────────────────────────────────

class RegimeGRU(nn.Module):
    """GRU with shared encoder + regime-specific output heads.

    Shared GRU encoder learns universal temporal patterns.
    Each regime has its own linear head that maps hidden state to prediction.
    This allows the model to adapt its output strategy per regime.
    """
    def __init__(self, input_dim, n_regimes, lag_idx, lag_init_scale=0.90,
                 hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.lag_idx = lag_idx
        self.n_regimes = n_regimes

        # Lag residual pathway
        self.lag_scale = nn.Parameter(torch.tensor(lag_init_scale))
        self.lag_bias = nn.Parameter(torch.tensor(0.0))

        # Shared GRU encoder
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)

        # Regime-specific heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(64, 1))
            for _ in range(n_regimes)
        ])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x, regime, lengths=None):
        """x: (batch, seq_len, features), regime: (batch, seq_len) int tensor."""
        lag_pred = self.lag_scale * x[:, :, self.lag_idx] + self.lag_bias

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x)

        # Apply regime-specific head per position
        B, T, H = out.shape
        out_flat = out.view(-1, H)  # (B*T, H)
        regime_flat = regime.view(-1)  # (B*T,)

        correction_flat = torch.zeros(B * T, device=out.device)
        for r in range(self.n_regimes):
            mask = regime_flat == r
            if mask.any():
                correction_flat[mask] = self.heads[r](out_flat[mask]).squeeze(-1)

        correction = correction_flat.view(B, T)
        return lag_pred + correction


# ─── Training Function ──────────────────────────────────────────────────────

def train_regime_gru(X_tr, y_tr, w_tr, regime_tr, X_va, y_va, w_va, regime_va,
                     date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                     feat_mean, feat_std, label="Regime GRU"):
    section(label)

    lag_col_indices = [ALL_FEATURES.index(c) for c in LAG_FEAT_COLS]
    lag_init = float(0.90 * feat_std[LAG_R6_IDX])

    torch.manual_seed(42)
    model = RegimeGRU(
        input_dim=len(ALL_FEATURES),
        n_regimes=N_REGIMES,
        lag_idx=LAG_R6_IDX,
        lag_init_scale=lag_init,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Regime heads: {N_REGIMES} (shared GRU encoder)")

    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=64)

    # Dataset with regime labels
    class RegimeDataset:
        def __init__(self, X, y, w, regime, chunks):
            self.X, self.y, self.w, self.regime = X, y, w, regime
            self.chunks = chunks

        def __len__(self):
            return len(self.chunks)

        def __getitem__(self, idx):
            s, e = self.chunks[idx]
            return self.X[s:e], self.y[s:e], self.w[s:e], self.regime[s:e], e - s

    def collate_regime(batch):
        xs, ys, ws, regimes, lens = zip(*batch)
        max_len = max(lens)
        B = len(batch)
        nf = xs[0].shape[1]

        X_pad = np.zeros((B, max_len, nf), dtype=np.float32)
        y_pad = np.zeros((B, max_len), dtype=np.float32)
        w_pad = np.zeros((B, max_len), dtype=np.float32)
        r_pad = np.zeros((B, max_len), dtype=np.int64)

        for i, (x, y, w, r, l) in enumerate(zip(xs, ys, ws, regimes, lens)):
            X_pad[i, :l] = x
            y_pad[i, :l] = y
            w_pad[i, :l] = w
            r_pad[i, :l] = r

        return (torch.from_numpy(X_pad), torch.from_numpy(y_pad),
                torch.from_numpy(w_pad), torch.from_numpy(r_pad),
                torch.tensor(lens, dtype=torch.long))

    dataset = RegimeDataset(X_tr, y_tr, w_tr, regime_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        collate_fn=collate_regime, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_r2, best_epoch, best_state = -np.inf, 0, None
    no_improve = 0

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b, w_b, r_b, lens in loader:
            X_b = X_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)
            w_b = w_b.to(DEVICE, non_blocking=True)
            r_b = r_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(X_b, r_b, lens)

            mask = torch.arange(pred.shape[1], device=DEVICE).unsqueeze(0) < lens.unsqueeze(1).to(DEVICE)
            loss = (w_b * (pred - y_b)**2 * mask.float()).sum() / w_b[mask].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate: need to compute regime for val data too
        model.eval()
        val_pred = predict_regime_gru(model, X_va, regime_va, date_ids_va, sym_ids_va)
        val_r2 = weighted_r2(y_va, val_pred, w_va)
        scheduler.step()

        print(f"  Epoch {ep:2d}: loss={total_loss/len(loader):.6f}  val_R²={val_r2:.6f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

        if val_r2 > best_r2:
            best_r2, best_epoch = val_r2, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {ep} (best: {best_epoch})")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.to(DEVICE)

    # Final metrics
    model.eval()
    with torch.no_grad():
        val_pred = predict_regime_gru(model, X_va, regime_va, date_ids_va, sym_ids_va)
    val_r2 = weighted_r2(y_va, val_pred, w_va)

    with torch.no_grad():
        tr_pred = predict_regime_gru(model, X_tr, regime_tr, date_ids_tr, sym_ids_tr)
    tr_r2 = weighted_r2(y_tr, tr_pred, w_tr)

    # Per-regime R²
    per_regime_r2 = {}
    for r in range(N_REGIMES):
        mask = regime_va == r
        if mask.sum() > 0:
            per_regime_r2[f"regime_{r}"] = weighted_r2(y_va[mask], val_pred[mask], w_va[mask])

    print(f"\n  RESULT: Val R² = {val_r2:.6f}  (train: {tr_r2:.6f}, gap: {tr_r2 - val_r2:.6f})")
    print(f"  vs Standard GRU+lag (0.883): {val_r2 - 0.883:+.6f}")
    print(f"  Per-regime R²: {per_regime_r2}")

    torch.save(best_state, MODEL_DIR / "gru_regime.pt")

    return {
        "val_r2": float(val_r2), "train_r2": float(tr_r2),
        "best_epoch": best_epoch, "train_time": train_time,
        "n_params": n_params,
        "per_regime_r2": {k: float(v) for k, v in per_regime_r2.items()},
    }


@torch.no_grad()
def predict_regime_gru(model, X, regime, date_ids, symbol_ids):
    """Predict using regime GRU, processing each (symbol, date) group."""
    model.eval()
    groups = find_sequence_groups(symbol_ids, date_ids)
    preds = np.zeros(len(X), dtype=np.float32)

    for g_start, g_end in groups:
        seq = torch.from_numpy(X[g_start:g_end]).unsqueeze(0).to(DEVICE)
        reg = torch.from_numpy(regime[g_start:g_end]).unsqueeze(0).to(DEVICE)
        out = model(seq, reg)
        preds[g_start:g_end] = out[0].cpu().numpy()

    return preds


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # ─── Load Data ─────────────────────────────────────────────────────────
    section("Loading data")
    X_tr, y_tr, w_tr, _, date_ids_tr, sym_ids_tr = load_pass(TRAIN_LO, TRAIN_HI)
    X_va, y_va, w_va, _, date_ids_va, sym_ids_va = load_pass(
        VAL_LO, VAL_HI, overlap_start=VAL_OVERLAP_START)

    section("Standardizing features")
    X_tr, X_va, feat_mean, feat_std = standardize(X_tr, X_va)
    lag_init = float(0.90 * feat_std[LAG_R6_IDX])
    print(f"  Train: {X_tr.shape}, Val: {X_va.shape}")

    # ─── Compute Regime Features ───────────────────────────────────────────
    section("Computing regime features")
    print("  Training set:")
    regime_tr, regime_feat_tr, regime_names = compute_regime_features(
        X_tr, date_ids_tr, date_ids_tr, sym_ids_tr, feat_mean, feat_std)

    print("\n  Validation set:")
    regime_va, regime_feat_va, _ = compute_regime_features(
        X_va, date_ids_va, date_ids_va, sym_ids_va, feat_mean, feat_std)

    # ─── Baseline: Standard GRU (no regime) ────────────────────────────────
    section("Baseline: Standard GRU (no regime)")
    torch.manual_seed(42)
    model_base = GRUModel(
        input_dim=len(ALL_FEATURES),
        lag_idx=LAG_R6_IDX,
        lag_init_scale=lag_init,
    ).to(DEVICE)

    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=64)

    dataset = ChunkedSeqDataset(X_tr, y_tr, w_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model_base.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_r2, best_epoch, best_state = -np.inf, 0, None
    no_improve = 0

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model_base.train()
        total_loss = 0.0
        for X_b, y_b, w_b, lens in loader:
            X_b = X_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)
            w_b = w_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model_base(X_b, lens)
            mask = torch.arange(pred.shape[1], device=DEVICE).unsqueeze(0) < lens.unsqueeze(1).to(DEVICE)
            loss = (w_b * (pred - y_b)**2 * mask.float()).sum() / w_b[mask].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_base.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_pred = predict_gru(model_base, X_va, date_ids_va, sym_ids_va)
        val_r2 = weighted_r2(y_va, val_pred, w_va)
        scheduler.step()

        if val_r2 > best_r2:
            best_r2, best_epoch = val_r2, ep
            best_state = {k: v.cpu().clone() for k, v in model_base.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    baseline_time = time.time() - t0
    model_base.load_state_dict(best_state)
    model_base.to(DEVICE)
    model_base.eval()
    with torch.no_grad():
        val_pred_base = predict_gru(model_base, X_va, date_ids_va, sym_ids_va)
    baseline_r2 = weighted_r2(y_va, val_pred_base, w_va)
    print(f"  Baseline GRU Val R² = {baseline_r2:.6f} ({baseline_time:.0f}s)")
    results["baseline_gru"] = {"val_r2": float(baseline_r2), "train_time": baseline_time}

    # ─── Regime-Conditional GRU ────────────────────────────────────────────
    res_regime = train_regime_gru(
        X_tr, y_tr, w_tr, regime_tr,
        X_va, y_va, w_va, regime_va,
        date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
        feat_mean, feat_std)
    results["regime_gru"] = res_regime

    # ─── Per-Regime Analysis ───────────────────────────────────────────────
    section("Per-Regime Analysis")

    # Compare baseline vs regime model per regime
    for r in range(N_REGIMES):
        mask = regime_va == r
        if mask.sum() == 0:
            continue
        r2_base_r = weighted_r2(y_va[mask], val_pred_base[mask], w_va[mask])
        r2_regime_r = results["regime_gru"]["per_regime_r2"].get(f"regime_{r}", 0)
        print(f"  {regime_names[r]}: baseline R²={r2_base_r:.6f}, regime R²={r2_regime_r:.6f}, "
              f"delta={r2_regime_r-r2_base_r:+.6f}, rows={mask.sum():,}")

    # ─── Summary ───────────────────────────────────────────────────────────
    section("REGIME MODELING RESULTS")

    print(f"\n  {'Model':<45} {'Val R²':>10} {'Δ vs baseline':>14}")
    print("  " + "-" * 71)
    print(f"  {'Standard GRU+lag':<43} {baseline_r2:10.6f} {'—':>14}")
    print(f"  {'Regime-conditional GRU':<43} {results['regime_gru']['val_r2']:10.6f} "
          f"{results['regime_gru']['val_r2']-baseline_r2:+14.6f}")

    if results["regime_gru"]["val_r2"] > baseline_r2 + 0.001:
        print(f"\n  ✓ Regime modeling adds value — market state matters for prediction.")
    else:
        print(f"\n  × Regime modeling does not improve over standard GRU.")
        print(f"  The lag relationship may be stable enough that regime splitting isn't needed.")

    rp = REPORT_DIR / "results.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {rp}")
