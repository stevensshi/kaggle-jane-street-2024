"""Phase 5: Neural Network models for Jane Street 2024.

Experiments:
  1. MLP baseline (no lag residual)
  2. MLP with lag residual (pred = scale * lag1_r6 + MLP correction)
  3. Multi-task MLP (primary: r6, auxiliary: r3, with lag residual)
  4. GRU with lag residual (per-symbol-per-day sequences)

Baseline to beat: LightGBM (59 features) Val R² = 0.856097

Training: dates 900-1188 (~10M rows)
Validation: dates 1189-1443 (~9.4M rows)
Same CV as Phases 3-4 for fair comparison.
"""

import gc
import time
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import weighted_r2

# ─── Constants ───────────────────────────────────────────────────────────────

TRAIN_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "data" / "raw" / "train.parquet"
)
assert Path(TRAIN_PATH).exists(), f"Data not found: {TRAIN_PATH}"

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "neural_network"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_54 = [
    "feature_06", "feature_61", "feature_30", "feature_36", "feature_07",
    "feature_04", "feature_24", "feature_31", "feature_29", "feature_20",
    "feature_22", "feature_21", "feature_58", "feature_08", "feature_47",
    "feature_01", "feature_25", "feature_28", "feature_05", "feature_60",
    "feature_26", "feature_23", "feature_38", "feature_72", "feature_69",
    "feature_27", "feature_59", "feature_33", "feature_52", "feature_15",
    "feature_49", "feature_66", "feature_56", "feature_14", "feature_37",
    "feature_34", "feature_17", "feature_42", "feature_70", "feature_77",
    "feature_45", "feature_13", "feature_12", "feature_53", "feature_67",
    "feature_78", "feature_11", "feature_57", "feature_19", "feature_65",
    "feature_10", "feature_16", "feature_09", "feature_18",
]

LAG_RESPONDERS = [6, 3, 7, 4, 0]
LAG_FEAT_COLS = [f"lag1_r{i}" for i in LAG_RESPONDERS]
ALL_FEATURES = SELECTED_54 + LAG_FEAT_COLS  # 59 total
LAG_R6_IDX = ALL_FEATURES.index("lag1_r6")  # index in feature vector

TARGET_COL = "responder_6"
AUX_TARGET = "responder_3"  # corr=0.73 with responder_6

TRAIN_LO, TRAIN_HI = 900, 1188
VAL_LO, VAL_HI = 1189, 1443
VAL_OVERLAP_START = 1174  # 15-day overlap for correct lag at boundary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
PATIENCE = 3

LGB_R2 = 0.856097
NAIVE_LAG_R2 = 0.811445

results = {}


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_pass(date_lo, date_hi, overlap_start=None):
    """Load one pass of data, engineer lag features, return numpy arrays.

    Two-pass approach (train then val) avoids OOM on 15 GB RAM.
    Val pass uses overlap dates so lag features are warm at the boundary.

    Returns: X, y, w, y_aux, date_ids, symbol_ids
    """
    actual_lo = overlap_start if overlap_start is not None else date_lo

    load_cols = list(set(
        ["date_id", "time_id", "symbol_id", "weight", TARGET_COL, AUX_TARGET]
        + SELECTED_54
        + [f"responder_{i}" for i in LAG_RESPONDERS]
    ))

    t0 = time.time()
    df = (
        pl.scan_parquet(TRAIN_PATH)
        .filter((pl.col("date_id") >= actual_lo) & (pl.col("date_id") <= date_hi))
        .select(load_cols)
        .collect(engine="streaming")
        .sort(["symbol_id", "date_id", "time_id"])
    )
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Lag responder features: shift(1) within each symbol
    df = df.with_columns([
        pl.col(f"responder_{i}").shift(1).over("symbol_id").alias(f"lag1_r{i}")
        for i in LAG_RESPONDERS
    ])

    # Discard overlap rows
    df = df.filter((pl.col("date_id") >= date_lo) & (pl.col("date_id") <= date_hi))

    n = len(df)
    nf = len(ALL_FEATURES)

    # Column-by-column extraction (memory efficient)
    X = np.empty((n, nf), dtype=np.float32)
    for i, col in enumerate(ALL_FEATURES):
        arr = df[col].to_numpy()
        np.nan_to_num(arr, copy=False)
        X[:, i] = arr
        del arr

    y = df[TARGET_COL].to_numpy().astype(np.float32)
    w = df["weight"].to_numpy().astype(np.float32)
    y_aux = df[AUX_TARGET].to_numpy().astype(np.float32)
    np.nan_to_num(y_aux, copy=False)
    date_ids = df["date_id"].to_numpy().astype(np.int32)
    symbol_ids = df["symbol_id"].to_numpy().astype(np.int32)

    del df
    gc.collect()
    print(f"  Extracted X={X.shape} in {time.time()-t0:.1f}s total")
    return X, y, w, y_aux, date_ids, symbol_ids


def standardize(X_train, X_val):
    """Compute mean/std from train, apply to both. Returns normalized arrays + stats."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-7] = 1.0
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


# ─── Model Definitions ──────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple feedforward MLP."""
    def __init__(self, input_dim, hidden=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPLagResidual(nn.Module):
    """MLP with lag residual: pred = lag_scale * lag1_r6 + lag_bias + MLP(x).

    The lag pathway handles the dominant signal (~73% of LightGBM gain).
    The MLP learns corrections. Last layer initialized near zero so the
    model starts with pure lag prediction and gradually adds corrections.
    """
    def __init__(self, input_dim, lag_idx, lag_init_scale=0.90,
                 hidden=(256, 256, 128), dropout=0.1):
        super().__init__()
        self.lag_idx = lag_idx
        self.lag_scale = nn.Parameter(torch.tensor(lag_init_scale))
        self.lag_bias = nn.Parameter(torch.tensor(0.0))

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.correction = nn.Sequential(*layers)

    def forward(self, x):
        lag_pred = self.lag_scale * x[:, self.lag_idx] + self.lag_bias
        return lag_pred + self.correction(x).squeeze(-1)


class MultiTaskMLP(nn.Module):
    """Multi-task MLP: shared encoder + per-task heads, lag residual on primary.

    Auxiliary task (responder_3, corr=0.73) encourages the shared encoder
    to learn richer representations.
    """
    def __init__(self, input_dim, lag_idx, lag_init_scale=0.90,
                 hidden=(256, 256), head_dim=128, dropout=0.1):
        super().__init__()
        self.lag_idx = lag_idx
        self.lag_scale = nn.Parameter(torch.tensor(lag_init_scale))
        self.lag_bias = nn.Parameter(torch.tensor(0.0))

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)

        self.head_primary = nn.Sequential(
            nn.Linear(prev, head_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(head_dim, 1))
        nn.init.zeros_(self.head_primary[-1].weight)
        nn.init.zeros_(self.head_primary[-1].bias)

        self.head_aux = nn.Sequential(
            nn.Linear(prev, head_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(head_dim, 1))

    def forward(self, x):
        lag_pred = self.lag_scale * x[:, self.lag_idx] + self.lag_bias
        shared = self.encoder(x)
        primary = lag_pred + self.head_primary(shared).squeeze(-1)
        aux = self.head_aux(shared).squeeze(-1)
        return primary, aux


class GRUModel(nn.Module):
    """GRU with lag residual for sequence-to-sequence prediction.

    Processes per-symbol-per-day sequences. Each time step produces a prediction.
    The lag residual provides a strong prior; GRU learns temporal corrections.
    """
    def __init__(self, input_dim, lag_idx, lag_init_scale=0.90,
                 hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.lag_idx = lag_idx
        self.lag_scale = nn.Parameter(torch.tensor(lag_init_scale))
        self.lag_bias = nn.Parameter(torch.tensor(0.0))

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers,
                          batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, lengths=None):
        """x: (batch, seq_len, features) -> (batch, seq_len) predictions."""
        lag_pred = self.lag_scale * x[:, :, self.lag_idx] + self.lag_bias

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x)

        return lag_pred + self.head(out).squeeze(-1)


# ─── Training Utilities ─────────────────────────────────────────────────────

def make_loader(X, y, w, y_aux=None, batch_size=BATCH_SIZE, shuffle=True):
    tensors = [torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(w)]
    if y_aux is not None:
        tensors.append(torch.from_numpy(y_aux))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                      shuffle=shuffle, pin_memory=True, num_workers=0)


def train_epoch_mlp(model, loader, optimizer, multi_task=False, aux_weight=0.3):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = [t.to(DEVICE, non_blocking=True) for t in batch]
        optimizer.zero_grad(set_to_none=True)

        if multi_task:
            x, y_p, w, y_a = batch
            pred_p, pred_a = model(x)
            loss = (w * (pred_p - y_p)**2).mean() + aux_weight * (w * (pred_a - y_a)**2).mean()
        else:
            x, y, w = batch[:3]
            loss = (w * (model(x) - y)**2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def predict_mlp(model, X, batch_size=BATCH_SIZE * 4, multi_task=False):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
        out = model(xb)[0] if multi_task else model(xb)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds)


def train_mlp_experiment(model, X_tr, y_tr, w_tr, X_va, y_va, w_va,
                         y_aux_tr=None, multi_task=False, aux_weight=0.3, label=""):
    """Full MLP training loop with early stopping on val R²."""
    section(label)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    loader = make_loader(X_tr, y_tr, w_tr, y_aux_tr if multi_task else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_r2, best_epoch, best_state = -np.inf, 0, None
    no_improve = 0

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        t_ep = time.time()
        loss = train_epoch_mlp(model, loader, optimizer, multi_task, aux_weight)
        val_pred = predict_mlp(model, X_va, multi_task=multi_task)
        val_r2 = weighted_r2(y_va, val_pred, w_va)
        scheduler.step()

        print(f"  Epoch {ep:2d}: loss={loss:.6f}  val_R²={val_r2:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.1e}  ({time.time()-t_ep:.0f}s)", flush=True)

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
    val_pred = predict_mlp(model, X_va, multi_task=multi_task)
    val_r2 = weighted_r2(y_va, val_pred, w_va)
    tr_pred = predict_mlp(model, X_tr, multi_task=multi_task)
    tr_r2 = weighted_r2(y_tr, tr_pred, w_tr)

    # Inference speed (single row)
    x1 = torch.from_numpy(X_va[:1]).to(DEVICE)
    model.eval()
    torch.cuda.synchronize()
    t_inf = time.time()
    for _ in range(1000):
        model(x1)[0] if multi_task else model(x1)
    torch.cuda.synchronize()
    infer_ms = (time.time() - t_inf) / 1000 * 1000

    print(f"\n  RESULT: Val R² = {val_r2:.6f}  (train: {tr_r2:.6f}, gap: {tr_r2 - val_r2:.6f})")
    print(f"  vs LightGBM: {val_r2 - LGB_R2:+.6f}")
    print(f"  Best epoch: {best_epoch}, time: {train_time:.0f}s, infer: {infer_ms:.3f}ms")

    return model, best_state, {
        "val_r2": float(val_r2), "train_r2": float(tr_r2),
        "best_epoch": best_epoch, "train_time": train_time,
        "infer_ms": infer_ms, "n_params": n_params,
    }


# ─── GRU Utilities ───────────────────────────────────────────────────────────

def find_sequence_groups(symbol_ids, date_ids):
    """Find (start, end) indices for each (symbol, date) group.

    Data must be sorted by (symbol_id, date_id, time_id).
    """
    n = len(symbol_ids)
    groups = []
    start = 0
    for i in range(1, n):
        if symbol_ids[i] != symbol_ids[i - 1] or date_ids[i] != date_ids[i - 1]:
            groups.append((start, i))
            start = i
    groups.append((start, n))
    return groups


def create_chunks(groups, chunk_size=64, min_len=4):
    """Split (symbol, date) groups into fixed-size chunks for efficient batching."""
    chunks = []
    for g_start, g_end in groups:
        g_len = g_end - g_start
        for k in range(0, g_len, chunk_size):
            c_start = g_start + k
            c_end = min(g_start + k + chunk_size, g_end)
            if c_end - c_start >= min_len:
                chunks.append((c_start, c_end))
    return chunks


class ChunkedSeqDataset:
    def __init__(self, X, y, w, chunks):
        self.X, self.y, self.w = X, y, w
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        s, e = self.chunks[idx]
        return self.X[s:e], self.y[s:e], self.w[s:e], e - s


def collate_sequences(batch):
    """Pad variable-length sequences to max length in batch."""
    xs, ys, ws, lens = zip(*batch)
    max_len = max(lens)
    B = len(batch)
    nf = xs[0].shape[1]

    X_pad = np.zeros((B, max_len, nf), dtype=np.float32)
    y_pad = np.zeros((B, max_len), dtype=np.float32)
    w_pad = np.zeros((B, max_len), dtype=np.float32)

    for i, (x, y, w, l) in enumerate(zip(xs, ys, ws, lens)):
        X_pad[i, :l] = x
        y_pad[i, :l] = y
        w_pad[i, :l] = w

    return (torch.from_numpy(X_pad), torch.from_numpy(y_pad),
            torch.from_numpy(w_pad), torch.tensor(lens, dtype=torch.long))


def train_epoch_gru(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for X_b, y_b, w_b, lens in loader:
        X_b = X_b.to(DEVICE, non_blocking=True)
        y_b = y_b.to(DEVICE, non_blocking=True)
        w_b = w_b.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(X_b, lens)

        # Only compute loss on non-padded positions
        mask = torch.arange(pred.shape[1], device=DEVICE).unsqueeze(0) < lens.unsqueeze(1).to(DEVICE)
        loss = (w_b * (pred - y_b)**2 * mask.float()).sum() / w_b[mask].sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def predict_gru(model, X, date_ids, symbol_ids):
    """Predict by processing each (symbol, date) group as one sequence."""
    model.eval()
    groups = find_sequence_groups(symbol_ids, date_ids)
    preds = np.zeros(len(X), dtype=np.float32)

    for g_start, g_end in groups:
        seq = torch.from_numpy(X[g_start:g_end]).unsqueeze(0).to(DEVICE)
        out = model(seq)
        preds[g_start:g_end] = out[0].cpu().numpy()

    return preds


def train_gru_experiment(model, X_tr, y_tr, w_tr, X_va, y_va, w_va,
                         date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                         chunk_size=64, seq_batch_size=128, label=""):
    """Full GRU training loop with chunked sequences."""
    section(label)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=chunk_size)
    print(f"  Training: {len(groups_tr):,} groups -> {len(chunks_tr):,} chunks (size={chunk_size})")

    dataset = ChunkedSeqDataset(X_tr, y_tr, w_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=seq_batch_size, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_r2, best_epoch, best_state = -np.inf, 0, None
    no_improve = 0

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        t_ep = time.time()
        loss = train_epoch_gru(model, loader, optimizer)

        val_pred = predict_gru(model, X_va, date_ids_va, sym_ids_va)
        val_r2 = weighted_r2(y_va, val_pred, w_va)
        scheduler.step()

        print(f"  Epoch {ep:2d}: loss={loss:.6f}  val_R²={val_r2:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.1e}  ({time.time()-t_ep:.0f}s)", flush=True)

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

    val_pred = predict_gru(model, X_va, date_ids_va, sym_ids_va)
    val_r2 = weighted_r2(y_va, val_pred, w_va)
    tr_pred = predict_gru(model, X_tr, date_ids_tr, sym_ids_tr)
    tr_r2 = weighted_r2(y_tr, tr_pred, w_tr)

    # Inference speed: single time step with hidden state
    x1 = torch.from_numpy(X_va[:1]).unsqueeze(0).to(DEVICE)  # (1, 1, features)
    model.eval()
    torch.cuda.synchronize()
    t_inf = time.time()
    for _ in range(1000):
        model(x1)
    torch.cuda.synchronize()
    infer_ms = (time.time() - t_inf) / 1000 * 1000

    print(f"\n  RESULT: Val R² = {val_r2:.6f}  (train: {tr_r2:.6f}, gap: {tr_r2 - val_r2:.6f})")
    print(f"  vs LightGBM: {val_r2 - LGB_R2:+.6f}")
    print(f"  Best epoch: {best_epoch}, time: {train_time:.0f}s, infer: {infer_ms:.3f}ms/step")

    return model, best_state, {
        "val_r2": float(val_r2), "train_r2": float(tr_r2),
        "best_epoch": best_epoch, "train_time": train_time,
        "infer_ms": infer_ms, "n_params": n_params,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ─── Load Data ───────────────────────────────────────────────────────────

    section("Pass 1: Training data (dates 900-1188)")
    X_tr, y_tr, w_tr, y_aux_tr, date_ids_tr, sym_ids_tr = load_pass(TRAIN_LO, TRAIN_HI)

    section("Pass 2: Validation data (dates 1189-1443, with overlap)")
    X_va, y_va, w_va, y_aux_va, date_ids_va, sym_ids_va = load_pass(
        VAL_LO, VAL_HI, overlap_start=VAL_OVERLAP_START)

    section("Standardizing features")
    X_tr, X_va, feat_mean, feat_std = standardize(X_tr, X_va)
    print(f"  Train: {X_tr.shape}, Val: {X_va.shape}")
    print(f"  lag1_r6 at index {LAG_R6_IDX}, std before normalization: {feat_std[LAG_R6_IDX]:.4f}")

    # Compute proper lag_scale initialization after standardization
    lag_init = float(0.90 * feat_std[LAG_R6_IDX])
    print(f"  Lag residual init scale (adjusted for normalization): {lag_init:.4f}")

    # Save normalization stats for inference
    np.savez(MODEL_DIR / "norm_stats.npz", mean=feat_mean, std=feat_std,
             features=ALL_FEATURES)

    # ─── Experiment 1: MLP Baseline ──────────────────────────────────────────

    torch.manual_seed(42)
    model1 = MLP(input_dim=len(ALL_FEATURES))
    _, state1, res1 = train_mlp_experiment(
        model1, X_tr, y_tr, w_tr, X_va, y_va, w_va,
        label="Exp 1: MLP Baseline (no lag residual)")
    results["mlp_baseline"] = res1
    torch.save(state1, MODEL_DIR / "mlp_baseline.pt")
    del model1, state1
    torch.cuda.empty_cache(); gc.collect()

    # ─── Experiment 2: MLP with Lag Residual ─────────────────────────────────

    torch.manual_seed(42)
    model2 = MLPLagResidual(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                            lag_init_scale=lag_init)
    _, state2, res2 = train_mlp_experiment(
        model2, X_tr, y_tr, w_tr, X_va, y_va, w_va,
        label="Exp 2: MLP + Lag Residual")
    results["mlp_lag_residual"] = res2
    torch.save(state2, MODEL_DIR / "mlp_lag_residual.pt")
    del model2, state2
    torch.cuda.empty_cache(); gc.collect()

    # ─── Experiment 3: Multi-task MLP ────────────────────────────────────────

    torch.manual_seed(42)
    model3 = MultiTaskMLP(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                          lag_init_scale=lag_init)
    _, state3, res3 = train_mlp_experiment(
        model3, X_tr, y_tr, w_tr, X_va, y_va, w_va,
        y_aux_tr=y_aux_tr, multi_task=True, aux_weight=0.3,
        label="Exp 3: Multi-task MLP (r6 + r3, aux_weight=0.3)")
    results["multitask_mlp"] = res3
    torch.save(state3, MODEL_DIR / "multitask_mlp.pt")
    del model3, state3
    torch.cuda.empty_cache(); gc.collect()

    # ─── Experiment 4: GRU with Lag Residual ─────────────────────────────────

    torch.manual_seed(42)
    model4 = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                      lag_init_scale=lag_init)
    _, state4, res4 = train_gru_experiment(
        model4, X_tr, y_tr, w_tr, X_va, y_va, w_va,
        date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
        label="Exp 4: GRU + Lag Residual")
    results["gru_lag_residual"] = res4
    torch.save(state4, MODEL_DIR / "gru_lag_residual.pt")
    del model4, state4
    torch.cuda.empty_cache(); gc.collect()

    # ─── Summary ─────────────────────────────────────────────────────────────

    section("RESULTS SUMMARY")

    print(f"\n{'Model':<45} {'Val R²':>10} {'Train R²':>10} {'Gap':>8} {'Time':>7} {'Infer':>8}")
    print("-" * 92)

    print(f"  {'Naive: predict lag_1 × 0.90':<43} {NAIVE_LAG_R2:10.6f}")
    print(f"  {'LightGBM (59 features, Phase 4)':<43} {LGB_R2:10.6f} {'0.868760':>10} {'0.0127':>8} {'322s':>7} {'0.111ms':>8}")
    print("-" * 92)

    for key, res in results.items():
        gap = res["train_r2"] - res["val_r2"]
        print(f"  {key:<43} {res['val_r2']:10.6f} {res['train_r2']:10.6f} "
              f"{gap:8.4f} {res['train_time']:6.0f}s {res['infer_ms']:.3f}ms")

    best_key = max(results, key=lambda k: results[k]["val_r2"])
    best_r2 = results[best_key]["val_r2"]
    print(f"\n  Best NN: {best_key} (R² = {best_r2:.6f}, vs LGB: {best_r2 - LGB_R2:+.6f})")

    rp = REPORT_DIR / "results.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {rp}")
    print(f"Models saved to {MODEL_DIR}/")
