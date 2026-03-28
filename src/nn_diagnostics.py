"""Phase 5 diagnostics: Is the GRU result (R²=0.883) legitimate?

Tests:
  1. Shuffled sequences: randomize time order within each (symbol, day).
     If GRU still scores ~0.883, temporal order doesn't matter → suspicious.
  2. Position-dependent R²: split predictions by position in sequence
     (early vs late). If only late positions are good → hidden state effect.
  3. MLP on same val data: verify MLP and GRU see identical data.
  4. Lag feature sanity: check lag1_r6 at position 0 in each sequence is NaN/0
     (no previous step exists).
"""

import gc
import time
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import weighted_r2
from neural_network import (
    TRAIN_PATH, ALL_FEATURES, LAG_R6_IDX, TARGET_COL, AUX_TARGET,
    SELECTED_54, LAG_RESPONDERS, LAG_FEAT_COLS,
    VAL_LO, VAL_HI, VAL_OVERLAP_START,
    DEVICE, GRUModel, MLPLagResidual,
    load_pass, standardize, predict_mlp, find_sequence_groups,
    TRAIN_LO, TRAIN_HI,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)


# ─── Load data ───────────────────────────────────────────────────────────────

section("Loading data")
X_tr, y_tr, w_tr, _, date_ids_tr, sym_ids_tr = load_pass(TRAIN_LO, TRAIN_HI)
X_va, y_va, w_va, _, date_ids_va, sym_ids_va = load_pass(
    VAL_LO, VAL_HI, overlap_start=VAL_OVERLAP_START)

X_tr_norm, X_va_norm, feat_mean, feat_std = standardize(X_tr, X_va)
del X_tr; gc.collect()

lag_init = float(0.90 * feat_std[LAG_R6_IDX])

# ─── Load trained GRU ───────────────────────────────────────────────────────

section("Loading trained GRU model")
gru = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX, lag_init_scale=lag_init)
gru.load_state_dict(torch.load(MODEL_DIR / "gru_lag_residual.pt", map_location="cpu"))
gru = gru.to(DEVICE)
gru.eval()
print(f"  GRU loaded, lag_scale = {gru.lag_scale.item():.6f}")

# Also load MLP for comparison
mlp = MLPLagResidual(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX, lag_init_scale=lag_init)
mlp.load_state_dict(torch.load(MODEL_DIR / "mlp_lag_residual.pt", map_location="cpu"))
mlp = mlp.to(DEVICE)
mlp.eval()
print(f"  MLP loaded, lag_scale = {mlp.lag_scale.item():.6f}")


# ─── Test 1: Lag feature sanity ─────────────────────────────────────────────

section("Test 1: Lag feature sanity check")

# Check raw (un-normalized) val data for lag1_r6 at position 0 of each sequence
groups_va = find_sequence_groups(sym_ids_va, date_ids_va)
lag_r6_raw_idx = ALL_FEATURES.index("lag1_r6")

# The raw (pre-normalization) lag1_r6 at sequence start should be ~0 (nan_to_num)
n_zero_start = 0
n_nonzero_start = 0
for g_start, g_end in groups_va:
    val = X_va[g_start, lag_r6_raw_idx]  # raw (un-normalized)
    if abs(val) < 1e-6:
        n_zero_start += 1
    else:
        n_nonzero_start += 1

print(f"  Sequences with lag1_r6 ≈ 0 at start: {n_zero_start}")
print(f"  Sequences with lag1_r6 ≠ 0 at start: {n_nonzero_start}")
print(f"  Total sequences: {len(groups_va)}")

if n_nonzero_start > n_zero_start:
    print("  ⚠ Most sequences have non-zero lag at position 0!")
    print("  This means lag carries over from previous day (within same symbol)")
    print("  This is correct behavior — data is sorted by (symbol, date, time)")
    print("  and shift(1).over('symbol_id') shifts across day boundaries.")


# ─── Test 2: GRU on original vs shuffled sequences ──────────────────────────

section("Test 2: Shuffled sequence test")

# Predict on original order
@torch.no_grad()
def predict_gru_groups(model, X, groups):
    model.eval()
    preds = np.zeros(len(X), dtype=np.float32)
    for g_start, g_end in groups:
        seq = torch.from_numpy(X[g_start:g_end]).unsqueeze(0).to(DEVICE)
        out = model(seq)
        preds[g_start:g_end] = out[0].cpu().numpy()
    return preds

print("  Predicting with original sequence order...")
preds_orig = predict_gru_groups(gru, X_va_norm, groups_va)
r2_orig = weighted_r2(y_va, preds_orig, w_va)
print(f"  GRU original order: R² = {r2_orig:.6f}")

# Predict with shuffled sequences (randomize time order within each group)
print("  Predicting with SHUFFLED sequence order...")
rng = np.random.RandomState(42)
X_va_shuffled = X_va_norm.copy()
y_va_shuffled = y_va.copy()
w_va_shuffled = w_va.copy()

for g_start, g_end in groups_va:
    idx = np.arange(g_start, g_end)
    rng.shuffle(idx)
    X_va_shuffled[g_start:g_end] = X_va_norm[idx]
    y_va_shuffled[g_start:g_end] = y_va[idx]
    w_va_shuffled[g_start:g_end] = w_va[idx]

preds_shuffled = predict_gru_groups(gru, X_va_shuffled, groups_va)
r2_shuffled = weighted_r2(y_va_shuffled, preds_shuffled, w_va_shuffled)
print(f"  GRU shuffled order: R² = {r2_shuffled:.6f}")
print(f"  Δ (orig - shuffled): {r2_orig - r2_shuffled:+.6f}")

if abs(r2_orig - r2_shuffled) < 0.002:
    print("  ⚠ SUSPICIOUS: GRU performs nearly identically on shuffled sequences!")
    print("  The temporal order doesn't seem to matter — GRU gain may not be from sequences.")
else:
    print("  ✓ GRU benefits significantly from temporal order — sequence modeling is real.")

del X_va_shuffled, y_va_shuffled, w_va_shuffled; gc.collect()


# ─── Test 3: Position-dependent R² ──────────────────────────────────────────

section("Test 3: Position-dependent R² (early vs late in sequence)")

positions = np.zeros(len(X_va_norm), dtype=np.int32)
seq_lengths = np.zeros(len(X_va_norm), dtype=np.int32)

for g_start, g_end in groups_va:
    g_len = g_end - g_start
    positions[g_start:g_end] = np.arange(g_len)
    seq_lengths[g_start:g_end] = g_len

# Split into quantiles by relative position (position / length)
rel_pos = positions / np.maximum(seq_lengths, 1)

# Also get MLP predictions for comparison
preds_mlp = predict_mlp(mlp, X_va_norm)
r2_mlp_total = weighted_r2(y_va, preds_mlp, w_va)
print(f"  MLP total R²: {r2_mlp_total:.6f}")
print(f"  GRU total R²: {r2_orig:.6f}")

print(f"\n  {'Segment':<25} {'N rows':>10} {'GRU R²':>10} {'MLP R²':>10} {'GRU-MLP':>10}")
print("  " + "-" * 67)

# Absolute position bins
pos_bins = [(0, 1, "pos 0 (first step)"),
            (1, 5, "pos 1-4"),
            (5, 20, "pos 5-19"),
            (20, 100, "pos 20-99"),
            (100, 500, "pos 100-499"),
            (500, 10000, "pos 500+")]

for lo, hi, label in pos_bins:
    mask = (positions >= lo) & (positions < hi)
    if mask.sum() < 100:
        continue
    y_m = y_va[mask]
    w_m = w_va[mask]
    r2_gru = weighted_r2(y_m, preds_orig[mask], w_m)
    r2_mlp = weighted_r2(y_m, preds_mlp[mask], w_m)
    delta = r2_gru - r2_mlp
    print(f"  {label:<25} {mask.sum():>10,} {r2_gru:10.6f} {r2_mlp:10.6f} {delta:+10.6f}")


# ─── Test 4: GRU vs MLP on first step only ──────────────────────────────────

section("Test 4: GRU hidden state = 0 (first step only)")

mask_first = positions == 0
r2_gru_first = weighted_r2(y_va[mask_first], preds_orig[mask_first], w_va[mask_first])
r2_mlp_first = weighted_r2(y_va[mask_first], preds_mlp[mask_first], w_va[mask_first])
print(f"  GRU at position 0 (no history): R² = {r2_gru_first:.6f}")
print(f"  MLP at position 0:              R² = {r2_mlp_first:.6f}")
print(f"  At position 0, GRU should ≈ MLP (no temporal context yet)")
delta_first = r2_gru_first - r2_mlp_first
if abs(delta_first) > 0.01:
    print(f"  ⚠ Large gap at position 0: {delta_first:+.6f} — something unexpected")


# ─── Test 5: Naive lag baseline on same data ─────────────────────────────────

section("Test 5: Naive lag baseline sanity check")
# Just predict lag_scale * lag1_r6 (using raw un-normalized data)
lag_raw = X_va[:, lag_r6_raw_idx]
for scale in [0.90, 0.85, 0.80, 1.00]:
    r2_naive = weighted_r2(y_va, scale * lag_raw, w_va)
    print(f"  Predict {scale:.2f} × lag1_r6: R² = {r2_naive:.6f}")


# ─── Summary ─────────────────────────────────────────────────────────────────

section("DIAGNOSTIC SUMMARY")
print(f"  MLP lag residual:        R² = {r2_mlp_total:.6f}")
print(f"  GRU original order:      R² = {r2_orig:.6f}")
print(f"  GRU shuffled order:      R² = {r2_shuffled:.6f}")
print(f"  GRU at position 0 only:  R² = {r2_gru_first:.6f}")
print(f"")
print(f"  Temporal order effect:   {r2_orig - r2_shuffled:+.6f}")
print(f"  GRU vs MLP at pos 0:    {r2_gru_first - r2_mlp_first:+.6f}")
print(f"  GRU total gain vs MLP:  {r2_orig - r2_mlp_total:+.6f}")
