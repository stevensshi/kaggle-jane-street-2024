"""Walk-forward cross-validation for Jane Street 2024.

Uses ALL available data (dates 0-1698) with an expanding-window scheme
to measure temporal stability and quantify lag dependency.

Each fold trains on [0, train_hi] and validates on [val_lo, val_hi].
Each fold runs twice: with lag features and without (lag columns zeroed)
to measure how much of the R² comes from lag1_r6 autocorrelation.

Usage:
    python walk_forward_cv.py              # all 5 folds, both variants
    python walk_forward_cv.py --folds 3,4,5  # specific folds only
    python walk_forward_cv.py --skip-no-lag  # skip the no-lag ablation

Outputs:
    reports/walk_forward_cv/results.json
    reports/walk_forward_cv/FINDINGS.md
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Fix TRAIN_PATH before importing neural_network (it uses 5-level parent by mistake)
import neural_network
neural_network.TRAIN_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "raw" / "train.parquet"
)
assert Path(neural_network.TRAIN_PATH).exists(), \
    f"Training data not found: {neural_network.TRAIN_PATH}"

from neural_network import (
    ALL_FEATURES, LAG_R6_IDX, LAG_FEAT_COLS,
    DEVICE, GRUModel,
    load_pass, standardize, predict_gru,
    find_sequence_groups, create_chunks, ChunkedSeqDataset, collate_sequences,
    train_epoch_gru,
    LR, WEIGHT_DECAY, EPOCHS, PATIENCE,
)
from evaluate import weighted_r2

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "walk_forward_cv"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LAG_OVERLAP = 15   # days of overlap before val_lo for lag warmup
SEQ_BATCH_SIZE = 128
CHUNK_SIZE = 64
SEED = 42

# ─── Fold Configuration ───────────────────────────────────────────────────────
# Expanding-window: train always starts at 0, val windows are non-overlapping.
# Holdout (1551-1698) is never touched during fold selection.

ALL_FOLDS = [
    {"fold": 1, "train_lo": 0, "train_hi": 500,  "val_lo": 501,  "val_hi": 700},
    {"fold": 2, "train_lo": 0, "train_hi": 700,  "val_lo": 701,  "val_hi": 900},
    {"fold": 3, "train_lo": 0, "train_hi": 900,  "val_lo": 901,  "val_hi": 1100},
    {"fold": 4, "train_lo": 0, "train_hi": 1100, "val_lo": 1101, "val_hi": 1350},
    {"fold": 5, "train_lo": 0, "train_hi": 1350, "val_lo": 1351, "val_hi": 1550},
]
HOLDOUT_LO, HOLDOUT_HI = 1551, 1698


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)


# ─── Per-date R² and lag autocorrelation analysis ────────────────────────────

def per_date_r2(y, preds, weights, date_ids):
    result = {}
    for d in sorted(set(date_ids)):
        mask = date_ids == d
        if mask.sum() > 0:
            result[int(d)] = float(weighted_r2(y[mask], preds[mask], weights[mask]))
    return result


def lag_autocorr_per_date(X_norm, y, date_ids, feat_mean, feat_std):
    """Compute corr(lag1_r6_raw, responder_6) per date."""
    lag_raw = X_norm[:, LAG_R6_IDX] * feat_std[LAG_R6_IDX] + feat_mean[LAG_R6_IDX]
    result = {}
    for d in sorted(set(date_ids)):
        mask = date_ids == d
        lv, yv = lag_raw[mask], y[mask]
        valid = ~(np.isnan(lv) | np.isnan(yv)) & (np.abs(lv) > 1e-12)
        if valid.sum() > 50:
            result[int(d)] = float(np.corrcoef(lv[valid], yv[valid])[0, 1])
    return result


# ─── Single fold training + evaluation ───────────────────────────────────────

def run_fold(train_lo, train_hi, val_lo, val_hi, fold_idx,
             include_lag=True, seed=SEED):
    tag = "with_lag" if include_lag else "no_lag"
    section(f"Fold {fold_idx} [{tag}] | train {train_lo}-{train_hi} | val {val_lo}-{val_hi}")

    # 1. Load data
    print(f"  Loading training data (dates {train_lo}-{train_hi})...")
    X_tr, y_tr, w_tr, _, date_ids_tr, sym_ids_tr = load_pass(train_lo, train_hi)

    print(f"  Loading validation data (dates {val_lo}-{val_hi}, overlap={val_lo - LAG_OVERLAP})...")
    X_va, y_va, w_va, _, date_ids_va, sym_ids_va = load_pass(
        val_lo, val_hi, overlap_start=val_lo - LAG_OVERLAP)

    n_train_rows = len(y_tr)
    n_val_rows = len(y_va)

    # 2. Standardize (stats from training only)
    X_tr, X_va, feat_mean, feat_std = standardize(X_tr, X_va)

    # 3. Lag ablation: zero out all 5 lag feature columns
    lag_col_indices = [ALL_FEATURES.index(c) for c in LAG_FEAT_COLS]
    if not include_lag:
        X_tr[:, lag_col_indices] = 0.0
        X_va[:, lag_col_indices] = 0.0

    # 4. Lag autocorrelation on val set (measured before zeroing to get true corr)
    if include_lag:
        lag_autocorr = lag_autocorr_per_date(X_va, y_va, date_ids_va, feat_mean, feat_std)
        mean_lag_autocorr = float(np.mean(list(lag_autocorr.values()))) if lag_autocorr else 0.0
    else:
        lag_autocorr = {}
        mean_lag_autocorr = 0.0

    # 5. Lag init scale (0 when ablating lag)
    lag_init = float(0.90 * feat_std[LAG_R6_IDX]) if include_lag else 0.0

    # 6. Build GRU model
    torch.manual_seed(seed)
    model = GRUModel(
        input_dim=len(ALL_FEATURES),
        lag_idx=LAG_R6_IDX,
        lag_init_scale=lag_init,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GRU params: {n_params:,} | lag_init={lag_init:.4f}")

    # 7. Build data loader
    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=CHUNK_SIZE)
    print(f"  {len(groups_tr):,} groups → {len(chunks_tr):,} chunks")

    dataset = ChunkedSeqDataset(X_tr, y_tr, w_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=SEQ_BATCH_SIZE, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0, pin_memory=True)

    # 8. Train with early stopping
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_r2, best_epoch, best_state = -np.inf, 0, None
    no_improve = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        loss = train_epoch_gru(model, loader, optimizer)
        val_pred = predict_gru(model, X_va, date_ids_va, sym_ids_va)
        val_r2 = weighted_r2(y_va, val_pred, w_va)
        scheduler.step()

        print(f"  Epoch {ep:2d}: loss={loss:.6f}  val_R²={val_r2:.6f}  "
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

    # 9. Final metrics with best state
    model.load_state_dict(best_state)
    model.to(DEVICE)

    val_pred = predict_gru(model, X_va, date_ids_va, sym_ids_va)
    val_r2 = float(weighted_r2(y_va, val_pred, w_va))

    tr_pred = predict_gru(model, X_tr, date_ids_tr, sym_ids_tr)
    tr_r2 = float(weighted_r2(y_tr, tr_pred, w_tr))

    # Per-date R²
    pdr2 = per_date_r2(y_va, val_pred, w_va, date_ids_va)

    print(f"\n  RESULT: val_R²={val_r2:.6f}  train_R²={tr_r2:.6f}  "
          f"gap={tr_r2-val_r2:.6f}  ({train_time:.0f}s)")
    if include_lag:
        print(f"  Mean lag autocorr (val): {mean_lag_autocorr:.4f}")

    # 10. Cleanup
    del model, X_tr, X_va, y_tr, w_tr, y_va, w_va, best_state
    del val_pred, tr_pred, dataset, loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "fold": fold_idx,
        "include_lag": include_lag,
        "train_dates": f"{train_lo}-{train_hi}",
        "val_dates": f"{val_lo}-{val_hi}",
        "n_train_rows": n_train_rows,
        "n_val_rows": n_val_rows,
        "val_r2": val_r2,
        "train_r2": tr_r2,
        "overfit_gap": tr_r2 - val_r2,
        "best_epoch": best_epoch,
        "train_time_s": train_time,
        "mean_lag_autocorr": mean_lag_autocorr,
        "per_date_r2": pdr2,
        "lag_autocorr_per_date": lag_autocorr,
    }


# ─── Aggregate and report ─────────────────────────────────────────────────────

def summarize(fold_results):
    r2s = [f["val_r2"] for f in fold_results]
    return {
        "mean_r2": float(np.mean(r2s)),
        "std_r2": float(np.std(r2s)),
        "min_r2": float(np.min(r2s)),
        "max_r2": float(np.max(r2s)),
        "spread": float(np.max(r2s) - np.min(r2s)),
    }


def write_findings(all_results):
    with_lag = all_results["folds_with_lag"]
    no_lag = all_results.get("folds_no_lag", [])
    sw = all_results.get("summary_with_lag", {})
    sn = all_results.get("summary_no_lag", {})
    dep = all_results.get("lag_dependency", {})

    lines = [
        "# Walk-Forward Cross-Validation Findings",
        "",
        f"Data range: dates 0-1698 (full dataset). Holdout {HOLDOUT_LO}-{HOLDOUT_HI} not evaluated here.",
        "",
        "## Summary Table",
        "",
        f"| Fold | Train Dates | Val Dates | Train Rows | With-Lag R² | No-Lag R² | Delta | Lag Autocorr |",
        f"|------|-------------|-----------|------------|-------------|-----------|-------|--------------|",
    ]

    no_lag_map = {f["fold"]: f for f in no_lag}
    for f in with_lag:
        nl = no_lag_map.get(f["fold"])
        nl_r2 = f"{nl['val_r2']:.6f}" if nl else "—"
        delta = f"{f['val_r2'] - nl['val_r2']:.6f}" if nl else "—"
        lines.append(
            f"| {f['fold']} | {f['train_dates']} | {f['val_dates']} | "
            f"{f['n_train_rows']:,} | {f['val_r2']:.6f} | {nl_r2} | {delta} | "
            f"{f['mean_lag_autocorr']:.4f} |"
        )

    lines += [
        "",
        "## Dispersion Analysis",
        "",
        f"**With lag:**  mean={sw.get('mean_r2', '?'):.6f}  std={sw.get('std_r2', '?'):.6f}  "
        f"[{sw.get('min_r2', '?'):.6f}, {sw.get('max_r2', '?'):.6f}]",
    ]
    if sn:
        lines.append(
            f"**No lag:**    mean={sn.get('mean_r2', '?'):.6f}  std={sn.get('std_r2', '?'):.6f}  "
            f"[{sn.get('min_r2', '?'):.6f}, {sn.get('max_r2', '?'):.6f}]"
        )

    if dep:
        lines += [
            "",
            "## Lag Dependency",
            "",
            f"Mean R² delta (with_lag − no_lag): **{dep.get('mean_delta', '?'):.6f}**",
            "",
            f"Per-fold deltas: {dep.get('per_fold_delta', [])}",
            "",
            "If delta ≈ 0.87 across all folds, ~99% of signal comes from lag1_r6.",
        ]

    lines += [
        "",
        "## Interpretation",
        "",
        "- **High std across folds** = temporal overfitting (R² depends on which regime is in val)",
        "- **Large lag delta** = model is lag-dominated; performance will collapse when lag autocorr changes",
        "- **Competition leaderboard #1 = 0.013890** vs our holdout 0.881 confirms this collapse in real future data",
        "",
        f"*Generated by src/walk_forward_cv.py*",
    ]

    with open(REPORT_DIR / "FINDINGS.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  FINDINGS written to {REPORT_DIR / 'FINDINGS.md'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward CV for Jane Street 2024")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold numbers to run, e.g. '3,4,5'")
    parser.add_argument("--skip-no-lag", action="store_true",
                        help="Skip the no-lag ablation variant")
    args = parser.parse_args()

    # Select folds
    if args.folds:
        requested = set(int(x) for x in args.folds.split(","))
        folds_to_run = [f for f in ALL_FOLDS if f["fold"] in requested]
    else:
        folds_to_run = ALL_FOLDS

    print(f"Device: {DEVICE}")
    print(f"Running folds: {[f['fold'] for f in folds_to_run]}")
    print(f"Lag ablation: {'SKIPPED' if args.skip_no_lag else 'enabled'}")
    print(f"Holdout ({HOLDOUT_LO}-{HOLDOUT_HI}) is NOT evaluated in this script.")

    all_results = {
        "folds_with_lag": [],
        "folds_no_lag": [],
        "holdout_dates": f"{HOLDOUT_LO}-{HOLDOUT_HI} (not evaluated here)",
    }

    # ── With lag ────────────────────────────────────────────────────────────
    for fold_cfg in folds_to_run:
        result = run_fold(
            train_lo=fold_cfg["train_lo"],
            train_hi=fold_cfg["train_hi"],
            val_lo=fold_cfg["val_lo"],
            val_hi=fold_cfg["val_hi"],
            fold_idx=fold_cfg["fold"],
            include_lag=True,
        )
        all_results["folds_with_lag"].append(result)
        # Save after each fold to preserve progress
        with open(REPORT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    all_results["summary_with_lag"] = summarize(all_results["folds_with_lag"])

    # ── No lag (ablation) ───────────────────────────────────────────────────
    if not args.skip_no_lag:
        for fold_cfg in folds_to_run:
            result = run_fold(
                train_lo=fold_cfg["train_lo"],
                train_hi=fold_cfg["train_hi"],
                val_lo=fold_cfg["val_lo"],
                val_hi=fold_cfg["val_hi"],
                fold_idx=fold_cfg["fold"],
                include_lag=False,
            )
            all_results["folds_no_lag"].append(result)
            with open(REPORT_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        all_results["summary_no_lag"] = summarize(all_results["folds_no_lag"])

        # Lag dependency analysis
        with_r2 = [f["val_r2"] for f in all_results["folds_with_lag"]]
        no_r2 = [f["val_r2"] for f in all_results["folds_no_lag"]]
        deltas = [w - n for w, n in zip(with_r2, no_r2)]
        all_results["lag_dependency"] = {
            "per_fold_delta": [round(d, 6) for d in deltas],
            "mean_delta": float(np.mean(deltas)),
        }

    # ── Final save + report ─────────────────────────────────────────────────
    with open(REPORT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {REPORT_DIR / 'results.json'}")

    write_findings(all_results)

    # Print summary
    print("\n" + "="*70)
    print("  WALK-FORWARD CV SUMMARY")
    print("="*70)
    if "summary_with_lag" in all_results:
        s = all_results["summary_with_lag"]
        print(f"  With lag:  mean={s['mean_r2']:.6f}  std={s['std_r2']:.6f}  "
              f"spread={s['spread']:.6f}")
    if "summary_no_lag" in all_results:
        s = all_results["summary_no_lag"]
        print(f"  No lag:    mean={s['mean_r2']:.6f}  std={s['std_r2']:.6f}  "
              f"spread={s['spread']:.6f}")
    if "lag_dependency" in all_results:
        print(f"  Lag delta: mean={all_results['lag_dependency']['mean_delta']:.6f}")
