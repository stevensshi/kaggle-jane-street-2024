"""Phase 6b: Residual Modeling + Feature-Only Baseline.

Motivation: The GRU+lag achieves R²≈0.88 on historical data, but ~99% of that
comes from lag1_r6 autocorrelation (r=0.90). When this signal collapses in
real future data, the model has almost nothing to fall back on
(feature-only LightGBM scores R²=0.011).

This script tests two critical hypotheses:

1. FEATURE-ONLY GRU: Zero out ALL lag features. If R² < 0.02, no architecture
   tweak will help when lag dies. This is the honest measure of feature signal.

2. RESIDUAL MODELING: Restructure training so the GRU predicts the RESIDUAL
   after lag baseline: residual = responder_6 - lag_scale × lag1_r6
   This forces the network to learn what lag CAN'T explain.
   At inference when lag weakens, the feature-based correction becomes primary.

3. LAG-DECAY SIMULATION: During training, randomly noise out lag features
   with probability p. This teaches the model to fall back on features when
   lag is unreliable. Acts as a regularizer against over-dependence on lag.

Usage:
    python improved_modeling.py --mode feature_only    # zero lag features
    python improved_modeling.py --mode residual         # predict residual
    python improved_modeling.py --mode lag_dropout      # random lag dropout
    python improved_modeling.py --mode all              # run all three

Baseline to beat: GRU+lag Val R² = 0.882852 (from Phase 5)
                  Feature-only LGB R² = 0.010782
"""

import argparse
import gc
import json
import time
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "improved_modeling"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

results = {}


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)


# ─── Model: GRU with Lag Dropout Regularization ───────────────────────────

class GRULagDropout(nn.Module):
    """GRU with lag feature dropout — randomly zeros lag features during training.

    This teaches the model to not depend entirely on lag. At inference time,
    lag features are passed through normally, but the model has learned to
    use features as a fallback.

    lag_dropout_prob=0.0 → behaves like standard GRU
    lag_dropout_prob=0.5 → 50% of training steps have lag features zeroed
    lag_dropout_prob=1.0 → always zero lag (feature-only model)
    """
    def __init__(self, input_dim, lag_indices, lag_init_scale=0.90,
                 lag_dropout_prob=0.0, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.lag_indices = lag_indices
        self.lag_dropout_prob = lag_dropout_prob

        # Lag residual pathway
        self.lag_r6_idx = lag_indices[0]  # lag1_r6 is first in lag_indices
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
        lag_pred = self.lag_scale * x[:, :, self.lag_r6_idx] + self.lag_bias

        # Apply lag dropout during training
        if self.training and self.lag_dropout_prob > 0:
            mask = torch.rand(x.shape[0], 1, len(self.lag_indices), device=x.device)
            mask = (mask > self.lag_dropout_prob).float()
            # Broadcast mask to all feature dims, but only apply to lag columns
            x_modified = x.clone()
            for i, lag_idx in enumerate(self.lag_indices):
                x_modified[:, :, lag_idx] *= mask[:, :, i]
        else:
            x_modified = x

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x_modified, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x_modified)

        correction = self.head(out).squeeze(-1)
        # Blend: when lag dropout is active, lag_pred is unreliable,
        # so the model learns to rely more on correction
        return lag_pred + correction


# ─── Model: Pure Residual GRU ─────────────────────────────────────────────

class GRUResidualOnly(nn.Module):
    """GRU that predicts ONLY the residual after lag baseline.

    Training target: residual = y - lag_scale × lag1_r6
    Prediction: pred = lag_scale × lag1_r6 + GRU(features)

    The GRU NEVER sees the lag features as input (they're zeroed).
    This forces it to learn corrections purely from the 54 base features.

    Key difference from GRULagDropout:
    - GRULagDropout: lag features present, randomly dropped
    - GRUResidualOnly: lag features NEVER used as input, only as output prior
    """
    def __init__(self, input_dim, lag_r6_idx, lag_init_scale=0.90,
                 hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.lag_r6_idx = lag_r6_idx
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
        lag_pred = self.lag_scale * x[:, :, self.lag_r6_idx] + self.lag_bias

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x)

        return lag_pred + self.head(out).squeeze(-1)


# ─── Training Functions ───────────────────────────────────────────────────

def train_feature_only_gru(X_tr, y_tr, w_tr, X_va, y_va, w_va,
                           date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                           feat_mean, feat_std, label="Feature-Only GRU"):
    """Train GRU with ALL lag features permanently zeroed.

    This measures the TRUE signal in features (no lag dependency).
    Expected R²: ~0.01 (same as feature-only LightGBM).
    If significantly higher, the GRU extracts additional temporal signal.
    """
    section(label)

    # Zero lag features in both train and val
    lag_col_indices = [ALL_FEATURES.index(c) for c in LAG_FEAT_COLS]
    X_tr_no_lag = X_tr.copy()
    X_va_no_lag = X_va.copy()
    X_tr_no_lag[:, lag_col_indices] = 0.0
    X_va_no_lag[:, lag_col_indices] = 0.0

    # Lag init = 0 since lag features are zero
    lag_init = 0.0

    torch.manual_seed(42)
    model = GRUResidualOnly(
        input_dim=len(ALL_FEATURES),
        lag_r6_idx=LAG_R6_IDX,
        lag_init_scale=lag_init,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} (lag_scale frozen to {lag_init})")

    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=64)
    print(f"  {len(groups_tr):,} groups → {len(chunks_tr):,} chunks")

    dataset = ChunkedSeqDataset(X_tr_no_lag, y_tr, w_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_r2, best_epoch, best_state = -np.inf, 0, None
    no_improve = 0

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b, w_b, lens in loader:
            X_b = X_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)
            w_b = w_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(X_b, lens)
            mask = torch.arange(pred.shape[1], device=DEVICE).unsqueeze(0) < lens.unsqueeze(1).to(DEVICE)
            loss = (w_b * (pred - y_b)**2 * mask.float()).sum() / w_b[mask].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_pred = predict_gru(model, X_va_no_lag, date_ids_va, sym_ids_va)
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
    val_pred = predict_gru(model, X_va_no_lag, date_ids_va, sym_ids_va)
    val_r2 = weighted_r2(y_va, val_pred, w_va)
    tr_pred = predict_gru(model, X_tr_no_lag, date_ids_tr, sym_ids_tr)
    tr_r2 = weighted_r2(y_tr, tr_pred, w_tr)

    print(f"\n  RESULT: Val R² = {val_r2:.6f}  (train: {tr_r2:.6f}, gap: {tr_r2 - val_r2:.6f})")
    print(f"  vs Feature-only LGB baseline (0.0108): {val_r2 - 0.0108:+.6f}")
    print(f"  vs Full GRU+lag (0.883): {val_r2 - 0.883:+.6f}")

    # Save model
    torch.save(best_state, MODEL_DIR / "gru_feature_only.pt")

    return {
        "val_r2": float(val_r2), "train_r2": float(tr_r2),
        "best_epoch": best_epoch, "train_time": train_time,
        "n_params": n_params,
        "interpretation": (
            "Features contain real temporal signal" if val_r2 > 0.05
            else "Features alone are nearly useless (expected for financial data)"
        ),
    }


def train_residual_gru(X_tr, y_tr, w_tr, X_va, y_va, w_va,
                       date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                       feat_mean, feat_std, lag_raw_va, label="Residual GRU"):
    """Train GRU to predict residual after lag baseline.

    Standard approach:
        loss = MSE(pred, y) where pred = lag_scale × lag1 + GRU(features)

    Residual approach:
        residual = y - 0.90 × lag1_r6
        loss = MSE(GRU(features_no_lag), residual)
        pred = 0.90 × lag1_r6 + GRU(features_no_lag)

    The GRU never sees lag features as input. It ONLY learns the correction.
    This is more honest about what the features can do.
    """
    section(label)

    # Zero lag features — GRU only sees base features
    lag_col_indices = [ALL_FEATURES.index(c) for c in LAG_FEAT_COLS]
    X_tr_no_lag = X_tr.copy()
    X_va_no_lag = X_va.copy()
    X_tr_no_lag[:, lag_col_indices] = 0.0
    X_va_no_lag[:, lag_col_indices] = 0.0

    # Compute residual targets
    lag_r6_raw = X_tr[:, LAG_R6_IDX] * feat_std[LAG_R6_IDX] + feat_mean[LAG_R6_IDX]
    lag_scale_fixed = 0.90
    y_tr_residual = y_tr - lag_scale_fixed * lag_r6_raw

    lag_r6_raw_va = X_va[:, LAG_R6_IDX] * feat_std[LAG_R6_IDX] + feat_mean[LAG_R6_IDX]
    y_va_residual = y_va - lag_scale_fixed * lag_r6_raw_va

    # For residual model, we predict residual directly (no lag_scale parameter)
    torch.manual_seed(42)

    # Simple GRU that outputs residual prediction
    class SimpleGRU(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.1):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=dropout if n_layers > 1 else 0)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1))

        def forward(self, x, lengths=None):
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                out, _ = self.gru(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            else:
                out, _ = self.gru(x)
            return self.head(out).squeeze(-1)

    model = SimpleGRU(input_dim=len(ALL_FEATURES)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Training target: residual = y - 0.90 × lag1_r6")
    print(f"  Residual std (train): {y_tr_residual.std():.6f}")
    print(f"  Residual std (val): {y_va_residual.std():.6f}")

    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=64)

    dataset = ChunkedSeqDataset(X_tr_no_lag, y_tr_residual, w_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_r2, best_epoch, best_state = -np.inf, 0, None
    no_improve = 0

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_b, res_b, w_b, lens in loader:
            X_b = X_b.to(DEVICE, non_blocking=True)
            res_b = res_b.to(DEVICE, non_blocking=True)
            w_b = w_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_res = model(X_b, lens)
            mask = torch.arange(pred_res.shape[1], device=DEVICE).unsqueeze(0) < lens.unsqueeze(1).to(DEVICE)
            loss = (w_b * (pred_res - res_b)**2 * mask.float()).sum() / w_b[mask].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on ORIGINAL target (not residual)
        model.eval()
        with torch.no_grad():
            res_pred = predict_gru(model, X_va_no_lag, date_ids_va, sym_ids_va)
        val_pred = lag_scale_fixed * lag_r6_raw_va + res_pred
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

    # Final metrics
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        res_pred_va = predict_gru(model, X_va_no_lag, date_ids_va, sym_ids_va)
    val_pred = lag_scale_fixed * lag_r6_raw_va + res_pred_va
    val_r2 = weighted_r2(y_va, val_pred, w_va)

    with torch.no_grad():
        res_pred_tr = predict_gru(model, X_tr_no_lag, date_ids_tr, sym_ids_tr)
    tr_pred = lag_scale_fixed * lag_r6_raw + res_pred_tr
    tr_r2 = weighted_r2(y_tr, tr_pred, w_tr)

    # Inference speed
    x1 = torch.from_numpy(X_va_no_lag[:1]).unsqueeze(0).to(DEVICE)
    torch.cuda.synchronize()
    t_inf = time.time()
    for _ in range(1000):
        model(x1)
    torch.cuda.synchronize()
    infer_ms = (time.time() - t_inf) / 1000 * 1000

    print(f"\n  RESULT: Val R² = {val_r2:.6f}  (train: {tr_r2:.6f}, gap: {tr_r2 - val_r2:.6f})")
    print(f"  vs Full GRU+lag (0.883): {val_r2 - 0.883:+.6f}")
    print(f"  vs Naive 0.90×lag (0.811): {val_r2 - 0.811:+.6f}")
    print(f"  Inference: {infer_ms:.3f}ms/step")

    torch.save(best_state, MODEL_DIR / "gru_residual_only.pt")

    return {
        "val_r2": float(val_r2), "train_r2": float(tr_r2),
        "best_epoch": best_epoch, "train_time": train_time,
        "infer_ms": infer_ms, "n_params": n_params,
        "residual_std_train": float(y_tr_residual.std()),
        "residual_std_val": float(y_va_residual.std()),
    }


def train_lag_dropout_gru(X_tr, y_tr, w_tr, X_va, y_va, w_va,
                          date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                          feat_mean, feat_std, dropout_prob=0.5,
                          label="Lag Dropout GRU"):
    """Train GRU with lag feature dropout.

    During training, lag features are randomly zeroed with probability `dropout_prob`.
    This forces the model to learn feature-based predictions as a fallback.

    dropout_prob=0.3: Mild — model still relies on lag most of the time
    dropout_prob=0.5: Medium — balanced lag/feature learning
    dropout_prob=0.7: Aggressive — model must learn features well
    """
    section(label)

    lag_col_indices = [ALL_FEATURES.index(c) for c in LAG_FEAT_COLS]
    lag_init = float(0.90 * feat_std[LAG_R6_IDX])

    torch.manual_seed(42)
    model = GRULagDropout(
        input_dim=len(ALL_FEATURES),
        lag_indices=lag_col_indices,
        lag_init_scale=lag_init,
        lag_dropout_prob=dropout_prob,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Lag dropout probability: {dropout_prob}")
    print(f"  lag_init_scale: {lag_init:.4f}")

    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=64)

    dataset = ChunkedSeqDataset(X_tr, y_tr, w_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_r2, best_epoch, best_state = -np.inf, 0, None
    no_improve = 0

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b, w_b, lens in loader:
            X_b = X_b.to(DEVICE, non_blocking=True)
            y_b = y_b.to(DEVICE, non_blocking=True)
            w_b = w_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(X_b, lens)
            mask = torch.arange(pred.shape[1], device=DEVICE).unsqueeze(0) < lens.unsqueeze(1).to(DEVICE)
            loss = (w_b * (pred - y_b)**2 * mask.float()).sum() / w_b[mask].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate at inference time (NO dropout — lag features passed through)
        val_pred = predict_gru(model, X_va, date_ids_va, sym_ids_va)
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
        val_pred = predict_gru(model, X_va, date_ids_va, sym_ids_va)
    val_r2 = weighted_r2(y_va, val_pred, w_va)

    with torch.no_grad():
        tr_pred = predict_gru(model, X_tr, date_ids_tr, sym_ids_tr)
    tr_r2 = weighted_r2(y_tr, tr_pred, w_tr)

    # Inference speed
    x1 = torch.from_numpy(X_va[:1]).unsqueeze(0).to(DEVICE)
    torch.cuda.synchronize()
    t_inf = time.time()
    for _ in range(1000):
        model(x1)
    torch.cuda.synchronize()
    infer_ms = (time.time() - t_inf) / 1000 * 1000

    print(f"\n  RESULT: Val R² = {val_r2:.6f}  (train: {tr_r2:.6f}, gap: {tr_r2 - val_r2:.6f})")
    print(f"  vs Full GRU+lag (0.883): {val_r2 - 0.883:+.6f}")
    print(f"  Inference: {infer_ms:.3f}ms/step")
    print(f"  Final lag_scale: {model.lag_scale.item():.6f}")

    torch.save(best_state, MODEL_DIR / f"gru_lag_dropout_{dropout_prob}.pt")

    return {
        "val_r2": float(val_r2), "train_r2": float(tr_r2),
        "best_epoch": best_epoch, "train_time": train_time,
        "infer_ms": infer_ms, "n_params": n_params,
        "dropout_prob": dropout_prob,
        "final_lag_scale": float(model.lag_scale.item()),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
                        choices=["feature_only", "residual", "lag_dropout", "all"])
    args = parser.parse_args()

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
    print(f"  lag_init_scale: {lag_init:.4f}")

    # Raw lag values for residual computation
    lag_r6_raw_va = X_va[:, LAG_R6_IDX] * feat_std[LAG_R6_IDX] + feat_mean[LAG_R6_IDX]

    # ─── Experiment: Feature-Only GRU ──────────────────────────────────────
    if args.mode in ("feature_only", "all"):
        res = train_feature_only_gru(
            X_tr, y_tr, w_tr, X_va, y_va, w_va,
            date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
            feat_mean, feat_std)
        results["feature_only_gru"] = res
        torch.cuda.empty_cache(); gc.collect()

    # ─── Experiment: Residual GRU ──────────────────────────────────────────
    if args.mode in ("residual", "all"):
        res = train_residual_gru(
            X_tr, y_tr, w_tr, X_va, y_va, w_va,
            date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
            feat_mean, feat_std, lag_r6_raw_va)
        results["residual_gru"] = res
        torch.cuda.empty_cache(); gc.collect()

    # ─── Experiment: Lag Dropout GRU ───────────────────────────────────────
    if args.mode in ("lag_dropout", "all"):
        for dp in [0.3, 0.5, 0.7]:
            res = train_lag_dropout_gru(
                X_tr, y_tr, w_tr, X_va, y_va, w_va,
                date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                feat_mean, feat_std, dropout_prob=dp,
                label=f"Lag Dropout GRU (p={dp})")
            results[f"lag_dropout_p{dp}"] = res
            torch.cuda.empty_cache(); gc.collect()

    # ─── Summary ───────────────────────────────────────────────────────────
    section("IMPROVED MODELING RESULTS")

    print(f"\n  {'Model':<45} {'Val R²':>10} {'Train R²':>10} {'Δ vs GRU+lag':>14}")
    print("  " + "-" * 81)
    print(f"  {'GRU + lag (Phase 5 baseline)':<43} {0.882852:10.6f} {0.868760:10.6f} {'—':>14}")
    print(f"  {'Naive 0.90×lag':<43} {0.811445:10.6f} {'—':>10} {'-0.071407':>14}")
    print(f"  {'Feature-only LightGBM (Phase 3)':<43} {0.010782:10.6f} {'—':>10} {'-0.872070':>14}")
    print("  " + "-" * 81)

    for key, res in sorted(results.items(), key=lambda x: x[1]["val_r2"], reverse=True):
        delta = res["val_r2"] - 0.882852
        tr = res.get("train_r2", 0.0)
        print(f"  {key:<43} {res['val_r2']:10.6f} {tr:10.6f} {delta:+14.6f}")

    # Key insight
    if "feature_only_gru" in results:
        fe_r2 = results["feature_only_gru"]["val_r2"]
        if fe_r2 > 0.05:
            print(f"\n  ⚠ Feature-only GRU scores R²={fe_r2:.4f} — features may contain temporal signal!")
            print(f"  This suggests the GRU extracts more from features than LightGBM can.")
        else:
            print(f"\n  ✓ Feature-only GRU scores R²={fe_r2:.4f} — confirms features alone are weak.")
            print(f"  When lag signal dies, we need a fundamentally different approach.")

    rp = REPORT_DIR / "results.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {rp}")
