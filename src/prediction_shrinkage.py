"""Phase 6d: Uncertainty Quantification & Prediction Shrinkage.

Motivation: When the lag signal is unreliable (low autocorrelation, regime shift),
the model should predict closer to zero rather than being confidently wrong.
This directly optimizes the competition metric — predicting zero scores R²=0,
which is better than a negative score from bad predictions.

Approach:
1. Use multi-seed ensemble variance as uncertainty estimate
2. Learn a shrinkage function: final_pred = shrink_factor × ensemble_pred
3. shrink_factor depends on: ensemble variance, lag reliability, regime
4. Optimize shrinkage on validation set

Usage:
    python prediction_shrinkage.py              # full analysis
"""

import gc
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import weighted_r2
from neural_network import (
    TRAIN_PATH, ALL_FEATURES, LAG_R6_IDX, TARGET_COL,
    SELECTED_54, LAG_RESPONDERS,
    TRAIN_LO, TRAIN_HI, VAL_LO, VAL_HI, VAL_OVERLAP_START,
    DEVICE, GRUModel,
    load_pass, standardize, predict_gru,
    find_sequence_groups, create_chunks, ChunkedSeqDataset, collate_sequences,
    BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, PATIENCE,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "prediction_shrinkage"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_SEEDS = [42, 123, 456]  # 3 seeds for efficiency
results = {}


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)


def train_single_gru(seed, X_tr, y_tr, w_tr, X_va, y_va, w_va,
                     date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                     lag_init):
    """Train one GRU model with given seed. Returns state_dict and metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GRUModel(
        input_dim=len(ALL_FEATURES),
        lag_idx=LAG_R6_IDX,
        lag_init_scale=lag_init,
    ).to(DEVICE)

    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=64)

    dataset = ChunkedSeqDataset(X_tr, y_tr, w_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_r2, best_epoch, best_state = -np.inf, 0, None
    no_improve = 0

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

        model.eval()
        with torch.no_grad():
            val_pred = predict_gru(model, X_va, date_ids_va, sym_ids_va)
        val_r2 = weighted_r2(y_va, val_pred, w_va)
        scheduler.step()

        if val_r2 > best_r2:
            best_r2, best_epoch = val_r2, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    del model
    torch.cuda.empty_cache()
    return best_state, best_r2, best_epoch


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

    # ─── Train Ensemble ────────────────────────────────────────────────────
    section(f"Training {len(ENSEMBLE_SEEDS)}-seed ensemble")
    states = []
    individual_r2s = []

    for seed in ENSEMBLE_SEEDS:
        print(f"  Training seed {seed}...")
        state, val_r2, best_ep = train_single_gru(
            seed, X_tr, y_tr, w_tr, X_va, y_va, w_va,
            date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
            lag_init)
        states.append(state)
        individual_r2s.append(val_r2)
        print(f"    Val R² = {val_r2:.6f} (epoch {best_ep})")

    # ─── Get Individual Predictions ────────────────────────────────────────
    section("Computing ensemble predictions")
    individual_preds = []
    for i, state in enumerate(states):
        model = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                         lag_init_scale=lag_init).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds = predict_gru(model, X_va, date_ids_va, sym_ids_va)
        individual_preds.append(preds)
        del model
        torch.cuda.empty_cache()

    individual_preds = np.array(individual_preds)  # (n_models, n_rows)

    # ─── Compute Uncertainty Statistics ────────────────────────────────────
    section("Uncertainty analysis")

    # Ensemble mean and variance
    ensemble_mean = individual_preds.mean(axis=0)
    ensemble_std = individual_preds.std(axis=0)

    # Standard ensemble prediction
    ensemble_r2 = weighted_r2(y_va, ensemble_mean, w_va)
    print(f"  Ensemble mean R²: {ensemble_r2:.6f}")
    print(f"  Individual R²s: {[f'{r:.6f}' for r in individual_r2s]}")
    print(f"  Ensemble std: mean={ensemble_std.mean():.6f}, "
          f"median={np.median(ensemble_std):.6f}, "
          f"p90={np.percentile(ensemble_std, 90):.6f}")

    # Correlation between ensemble std and prediction error
    errors = np.abs(ensemble_mean - y_va)
    corr_std_error = np.corrcoef(ensemble_std, errors)[0, 1]
    print(f"  Corr(ensemble_std, |error|): {corr_std_error:.4f}")
    if corr_std_error > 0.1:
        print(f"  ✓ Ensemble variance correlates with error — shrinkage should help")
    else:
        print(f"  × Weak correlation — ensemble variance may not be a good uncertainty signal")

    # ─── Shrinkage Analysis ───────────────────────────────────────────────
    section("Optimal constant shrinkage")

    best_shrink_r2 = -np.inf
    best_shrink_factor = 1.0

    for shrink in np.arange(0.50, 1.05, 0.01):
        shrunk = shrink * ensemble_mean
        r2 = weighted_r2(y_va, shrunk, w_va)
        if r2 > best_shrink_r2:
            best_shrink_r2 = r2
            best_shrink_factor = shrink

    print(f"  Optimal shrinkage factor: {best_shrink_factor:.2f}")
    print(f"  Shrunken R²: {best_shrink_r2:.6f}")
    print(f"  vs Unshrunk: {best_shrink_r2 - ensemble_r2:+.6f}")
    results["constant_shrinkage"] = {
        "optimal_factor": float(best_shrink_factor),
        "shrunk_r2": float(best_shrink_r2),
        "unshrunk_r2": float(ensemble_r2),
        "improvement": float(best_shrink_r2 - ensemble_r2),
    }

    # ─── Variance-Weighted Shrinkage ──────────────────────────────────────
    section("Variance-weighted shrinkage")

    # High-uncertainty predictions get shrunk more toward zero
    # shrink_factor = base_shrink × (1 - uncertainty_weight × normalized_std)
    # Try different uncertainty weights

    normalized_std = ensemble_std / (ensemble_std.mean() + 1e-8)

    for base_shrink in [0.90, 0.95, 1.0]:
        for unc_weight in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
            shrink_factors = base_shrink * (1.0 - unc_weight * normalized_std)
            shrunk = shrink_factors * ensemble_mean
            r2 = weighted_r2(y_va, shrunk, w_va)

            if unc_weight > 0 or base_shrink != 1.0:
                key = f"base{base_shrink:.2f}_unc{unc_weight:.2f}"
                if r2 > best_shrink_r2:
                    print(f"    {key}: R²={r2:.6f} (+{r2-ensemble_r2:+.6f}) **")
                elif r2 > ensemble_r2 + 0.0001:
                    print(f"    {key}: R²={r2:.6f} (+{r2-ensemble_r2:+.6f})")

                results[f"variance_shrink_{key}"] = {
                    "r2": float(r2), "base_shrink": base_shrink,
                    "unc_weight": unc_weight,
                }

    # ─── Lag-Reliability Shrinkage ────────────────────────────────────────
    section("Lag-reliability shrinkage")

    # When lag1_r6 is near zero, the lag signal is weak → shrink more
    lag_r6_raw = X_va[:, LAG_R6_IDX] * feat_std[LAG_R6_IDX] + feat_mean[LAG_R6_IDX]
    lag_abs = np.abs(lag_r6_raw)

    # Compute lag reliability per quantile of |lag1_r6|
    n_bins = 10
    lag_quantiles = np.linspace(0, 1, n_bins + 1)
    lag_bin_edges = np.quantile(lag_abs, lag_quantiles)

    print(f"  {'Lag |lag1_r6| Bin':<25} {'Rows':>8} {'R²':>10} {'Optimal Shrink':>15}")
    print("  " + "-" * 60)

    lag_reliability_shrink = np.ones(len(y_va), dtype=np.float32)

    for i in range(n_bins):
        lo, hi = lag_bin_edges[i], lag_bin_edges[i + 1]
        if lo == hi:
            continue
        mask = (lag_abs >= lo) & (lag_abs < hi)
        if mask.sum() < 100:
            continue

        y_bin = y_va[mask]
        w_bin = w_va[mask]
        p_bin = ensemble_mean[mask]

        r2_bin = weighted_r2(y_bin, p_bin, w_bin)

        # Find optimal shrink for this bin
        best_bin_shrink = 1.0
        best_bin_r2 = r2_bin
        for s in np.arange(0.50, 1.05, 0.01):
            r2_s = weighted_r2(y_bin, s * p_bin, w_bin)
            if r2_s > best_bin_r2:
                best_bin_r2 = r2_s
                best_bin_shrink = s

        label = f"[{lo:.3f}, {hi:.3f})"
        print(f"  {label:<25} {mask.sum():>8,} {r2_bin:>10.6f} {best_bin_shrink:>15.2f}")

        lag_reliability_shrink[mask] = best_bin_shrink

    # Apply lag-reliability shrinkage
    lag_shrunk = lag_reliability_shrink * ensemble_mean
    lag_shrink_r2 = weighted_r2(y_va, lag_shrunk, w_va)
    print(f"\n  Lag-reliability shrunk R²: {lag_shrink_r2:.6f} "
          f"(vs unshrunk: {lag_shrink_r2 - ensemble_r2:+.6f})")
    results["lag_reliability_shrinkage"] = {
        "r2": float(lag_shrink_r2),
        "improvement": float(lag_shrink_r2 - ensemble_r2),
    }

    # ─── Combined Shrinkage ───────────────────────────────────────────────
    section("Combined shrinkage (variance + lag reliability)")

    # Combine both: shrink more when BOTH ensemble variance is high AND lag is weak
    combined_shrink = lag_reliability_shrink * (1.0 - 0.1 * normalized_std)
    combined_shrunk = combined_shrink * ensemble_mean
    combined_r2 = weighted_r2(y_va, combined_shrunk, w_va)
    print(f"  Combined shrunk R²: {combined_r2:.6f} "
          f"(vs unshrunk: {combined_r2 - ensemble_r2:+.6f})")
    results["combined_shrinkage"] = {
        "r2": float(combined_r2),
        "improvement": float(combined_r2 - ensemble_r2),
    }

    # ─── Summary ───────────────────────────────────────────────────────────
    section("PREDICTION SHRINKAGE RESULTS")

    print(f"\n  {'Method':<50} {'Val R²':>10} {'Δ vs ensemble':>14}")
    print("  " + "-" * 76)
    print(f"  {'Ensemble mean (no shrinkage)':<48} {ensemble_r2:10.6f} {'—':>14}")

    # Sort by R² descending
    shrink_results = [(k, v) for k, v in results.items() if "r2" in v]
    shrink_results.sort(key=lambda x: x[1]["r2"], reverse=True)

    for key, res in shrink_results:
        delta = res["r2"] - ensemble_r2
        print(f"  {key:<48} {res['r2']:10.6f} {delta:+14.6f}")

    best_shrink_key = shrink_results[0][0] if shrink_results else None
    if best_shrink_key:
        print(f"\n  Best shrinkage: {best_shrink_key} "
              f"(R²={results[best_shrink_key]['r2']:.6f})")
        if results[best_shrink_key]["r2"] > ensemble_r2 + 0.001:
            print(f"  ✓ Shrinkage provides meaningful improvement")
        else:
            print(f"  × Shrinkage gains are marginal — ensemble is already well-calibrated")

    rp = REPORT_DIR / "results.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {rp}")
