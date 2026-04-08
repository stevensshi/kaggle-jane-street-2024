"""Phase 6: Online Learning — adapt GRU during inference.

Simulates the competition inference loop: step through validation data
one time step at a time, maintaining per-symbol GRU hidden states.
After each prediction, the revealed lagged responder provides a training
signal for online model adaptation.

Experiments:
  1. Frozen baseline (verify step-by-step matches batch R²)
  2. Online SGD, head-only, lr sweep
  3. Online SGD, all layers, lr sweep
  4. Day boundary: reset weights each day vs carry forward
  5. Lag-scale EMA tracking (non-gradient)
  6. Combined best

Base model: GRU lag-residual from Phase 5 (Val R² = 0.883)
"""

import copy
import gc
import json
import time
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import weighted_r2
from neural_network import (
    TRAIN_PATH, ALL_FEATURES, LAG_R6_IDX, TARGET_COL,
    SELECTED_54, LAG_RESPONDERS, LAG_FEAT_COLS,
    VAL_LO, VAL_HI, VAL_OVERLAP_START,
    TRAIN_LO, TRAIN_HI,
    DEVICE, GRUModel,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "online_learning"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_GRU_LAYERS = 2
HIDDEN_DIM = 128
MAX_SYMBOLS = 40  # competition has 39 symbols


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)


# ─── Data Loading for Online Simulation ──────────────────────────────────────

def load_online_data(date_lo, date_hi, overlap_start=None):
    """Load data for online simulation.

    Returns data sorted by (symbol_id, date_id, time_id) for lag engineering,
    plus time_ids needed for step-by-step processing.
    """
    actual_lo = overlap_start if overlap_start is not None else date_lo

    load_cols = list(set(
        ["date_id", "time_id", "symbol_id", "weight", TARGET_COL]
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

    # Lag features: shift(1) within each symbol
    df = df.with_columns([
        pl.col(f"responder_{i}").shift(1).over("symbol_id").alias(f"lag1_r{i}")
        for i in LAG_RESPONDERS
    ])

    # Discard overlap rows
    df = df.filter((pl.col("date_id") >= date_lo) & (pl.col("date_id") <= date_hi))

    # Re-sort by (date_id, time_id, symbol_id) for chronological stepping
    df = df.sort(["date_id", "time_id", "symbol_id"])

    n = len(df)
    nf = len(ALL_FEATURES)

    # Extract feature matrix
    X = np.empty((n, nf), dtype=np.float32)
    for i, col in enumerate(ALL_FEATURES):
        arr = df[col].to_numpy().copy()
        np.nan_to_num(arr, copy=False)
        X[:, i] = arr
        del arr

    y = df[TARGET_COL].to_numpy().astype(np.float32)
    w = df["weight"].to_numpy().astype(np.float32)
    date_ids = df["date_id"].to_numpy().astype(np.int32)
    time_ids = df["time_id"].to_numpy().astype(np.int32)
    symbol_ids = df["symbol_id"].to_numpy().astype(np.int32)

    del df; gc.collect()
    print(f"  X shape: {X.shape}, {time.time()-t0:.1f}s total")
    return X, y, w, date_ids, time_ids, symbol_ids


def build_time_step_index(date_ids, time_ids):
    """Build index of (start, end) for each unique (date_id, time_id)."""
    n = len(date_ids)
    groups = []
    start = 0
    for i in range(1, n):
        if date_ids[i] != date_ids[i - 1] or time_ids[i] != time_ids[i - 1]:
            groups.append((start, i))
            start = i
    groups.append((start, n))
    return groups


# ─── GRU Step-by-Step Inference ──────────────────────────────────────────────

def gru_step_batched(model, x_batch, h_indices, all_hidden):
    """Batched GRU forward for one time step.

    Args:
        model: GRUModel
        x_batch: (N, features) tensor on DEVICE
        h_indices: (N,) int tensor — symbol indices into all_hidden
        all_hidden: (n_layers, MAX_SYMBOLS, hidden_dim) tensor

    Returns:
        preds: (N,) tensor
        New hidden states are written in-place to all_hidden.
    """
    N = x_batch.shape[0]
    h = all_hidden[:, h_indices, :]           # (n_layers, N, hidden_dim)
    x_seq = x_batch.unsqueeze(1)              # (N, 1, features)
    out, h_new = model.gru(x_seq, h)          # out: (N, 1, hidden_dim)
    lag_pred = model.lag_scale * x_batch[:, model.lag_idx] + model.lag_bias
    correction = model.head(out.squeeze(1)).squeeze(-1)
    all_hidden[:, h_indices, :] = h_new
    return lag_pred + correction


def run_online_experiment(
    model_state,
    X_norm, y, w, date_ids, time_ids, symbol_ids,
    time_steps, lag_r6_raw,
    feat_mean, feat_std,
    update_mode="none",        # "none", "head", "all"
    lr=1e-4,
    reset_weights_daily=False,
    lag_ema_alpha=0.0,         # 0 = disabled, >0 = EMA tracking of lag_scale
    label="",
):
    """Run one online learning experiment.

    Args:
        model_state: state_dict to load (fresh copy each experiment)
        X_norm: (N, features) normalized features
        y: (N,) targets
        w: (N,) weights
        date_ids, time_ids, symbol_ids: (N,) arrays
        time_steps: list of (start, end) for each time step
        lag_r6_raw: (N,) un-normalized lag1_r6 values
        feat_mean, feat_std: normalization stats
        update_mode: "none", "head", "all"
        lr: learning rate for SGD updates
        reset_weights_daily: reset model to original weights each day
        lag_ema_alpha: EMA smoothing for lag_scale adaptation
        label: experiment name

    Returns:
        dict with val_r2, per_day_r2, etc.
    """
    section(label)

    # Fresh model
    lag_init = float(0.90 * feat_std[LAG_R6_IDX])
    model = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                     lag_init_scale=lag_init).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    original_state = copy.deepcopy(model_state) if reset_weights_daily else None

    # Set up optimizer for online updates
    optimizer = None
    if update_mode == "head":
        params = list(model.head.parameters()) + [model.lag_scale, model.lag_bias]
        optimizer = torch.optim.SGD(params, lr=lr)
    elif update_mode == "all":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Per-symbol hidden states
    all_hidden = torch.zeros(N_GRU_LAYERS, MAX_SYMBOLS, HIDDEN_DIM,
                             device=DEVICE, dtype=torch.float32)

    # Storage for online update replay
    prev_x = {}      # symbol_id -> normalized feature row (numpy)
    prev_h = {}      # symbol_id -> pre-step hidden state (n_layers, hidden_dim) tensor
    prev_weight = {}  # symbol_id -> weight

    # Lag EMA tracking
    ema_ratio = None  # tracked ratio of actual/predicted lag contribution

    # Output predictions
    all_preds = np.zeros(len(y), dtype=np.float32)

    # Per-day tracking
    day_results = {}
    current_day = None
    day_start_idx = 0
    n_updates = 0

    t0 = time.time()
    for step_i, (ts_start, ts_end) in enumerate(time_steps):
        day = date_ids[ts_start]
        syms = symbol_ids[ts_start:ts_end]
        n_syms = ts_end - ts_start

        # Day boundary handling
        if day != current_day:
            # Record previous day's R²
            if current_day is not None:
                day_mask = (date_ids[day_start_idx:ts_start] != -1)  # all True
                y_day = y[day_start_idx:ts_start]
                w_day = w[day_start_idx:ts_start]
                p_day = all_preds[day_start_idx:ts_start]
                if len(y_day) > 0:
                    day_results[int(current_day)] = weighted_r2(y_day, p_day, w_day)

            current_day = day
            day_start_idx = ts_start

            # Reset hidden states at day boundary
            all_hidden.zero_()
            prev_x.clear()
            prev_h.clear()
            prev_weight.clear()

            # Reset weights if configured
            if reset_weights_daily and original_state is not None:
                model.load_state_dict(original_state)
                model.eval()
                if update_mode == "head":
                    params = list(model.head.parameters()) + [model.lag_scale, model.lag_bias]
                    optimizer = torch.optim.SGD(params, lr=lr)
                elif update_mode == "all":
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # ─── Online update from previous step ────────────────────────
        if optimizer is not None and len(prev_x) > 0:
            # Find symbols present now that we also saw previously
            update_syms = []
            update_x_list = []
            update_true_y = []
            update_weights = []

            for i in range(n_syms):
                sid = int(syms[i])
                if sid in prev_x and sid in prev_h:
                    true_y_prev = lag_r6_raw[ts_start + i]
                    # Skip if lag is exactly 0 (likely NaN-filled first appearance)
                    if abs(true_y_prev) < 1e-12 and sid not in prev_weight:
                        continue
                    update_syms.append(sid)
                    update_x_list.append(prev_x[sid])
                    update_true_y.append(true_y_prev)
                    update_weights.append(prev_weight.get(sid, 1.0))

            if len(update_syms) >= 2:
                # Batch replay: re-run previous x through model with the
                # exact pre-step hidden state that produced the original prediction.
                x_replay = torch.from_numpy(
                    np.stack(update_x_list)).to(DEVICE)             # (K, features)
                true_y_t = torch.tensor(
                    update_true_y, dtype=torch.float32, device=DEVICE)
                weights_t = torch.tensor(
                    update_weights, dtype=torch.float32, device=DEVICE)

                # Stack saved pre-step hidden states: (n_layers, K, hidden_dim)
                h_replay = torch.stack(
                    [prev_h[sid] for sid in update_syms], dim=1).to(DEVICE)

                model.train()
                x_seq = x_replay.unsqueeze(1)  # (K, 1, features)
                out, _ = model.gru(x_seq, h_replay)
                lag_pred = model.lag_scale * x_replay[:, LAG_R6_IDX] + model.lag_bias
                correction = model.head(out.squeeze(1)).squeeze(-1)
                pred_replay = lag_pred + correction

                # Weighted loss — matches training objective and weighted_r2 metric
                loss = (weights_t * (pred_replay - true_y_t) ** 2).sum() / weights_t.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.eval()
                n_updates += 1

        # ─── Lag EMA adaptation ───────────────────────────────────────
        if lag_ema_alpha > 0 and len(prev_x) > 0:
            # Track ratio: actual_y / (lag_scale * normalized_lag)
            for i in range(n_syms):
                sid = int(syms[i])
                if sid in prev_x:
                    true_y_prev = lag_r6_raw[ts_start + i]
                    norm_lag_prev = prev_x[sid][LAG_R6_IDX]
                    if abs(norm_lag_prev) > 0.01:
                        ratio = true_y_prev / (norm_lag_prev * feat_std[LAG_R6_IDX]
                                               + feat_mean[LAG_R6_IDX] + 1e-8)
                        if ema_ratio is None:
                            ema_ratio = ratio
                        else:
                            ema_ratio = lag_ema_alpha * ratio + (1 - lag_ema_alpha) * ema_ratio
            # Apply: adjust lag_scale
            if ema_ratio is not None:
                with torch.no_grad():
                    target_scale = float(ema_ratio * feat_std[LAG_R6_IDX])
                    model.lag_scale.data.lerp_(
                        torch.tensor(target_scale, device=DEVICE, dtype=torch.float32),
                        lag_ema_alpha * 0.1)

        # ─── Forward pass (no gradient) ───────────────────────────────
        x_batch = torch.from_numpy(X_norm[ts_start:ts_end]).to(DEVICE)
        h_indices = torch.from_numpy(syms.astype(np.int64)).to(DEVICE)

        # Snapshot pre-step hidden states BEFORE gru_step_batched updates all_hidden
        h_snapshot = {int(syms[i]): all_hidden[:, int(syms[i]), :].detach().clone()
                      for i in range(n_syms)}

        with torch.no_grad():
            preds = gru_step_batched(model, x_batch, h_indices, all_hidden)
            all_preds[ts_start:ts_end] = preds.cpu().numpy()

        # Store for next step's online update
        prev_x.clear()
        prev_h.clear()
        prev_weight.clear()
        x_np = X_norm[ts_start:ts_end]
        for i in range(n_syms):
            sid = int(syms[i])
            prev_x[sid] = x_np[i].copy()
            prev_h[sid] = h_snapshot[sid]
            prev_weight[sid] = float(w[ts_start + i])

        # Progress
        if (step_i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step_i+1}/{len(time_steps)} ({elapsed:.0f}s)", flush=True)

    # Record last day
    if current_day is not None:
        y_day = y[day_start_idx:]
        w_day = w[day_start_idx:]
        p_day = all_preds[day_start_idx:]
        if len(y_day) > 0:
            day_results[int(current_day)] = weighted_r2(y_day, p_day, w_day)

    # Overall R²
    total_r2 = weighted_r2(y, all_preds, w)
    elapsed = time.time() - t0

    print(f"\n  Total Val R² = {total_r2:.6f}")
    print(f"  Online updates: {n_updates}")
    print(f"  Time: {elapsed:.1f}s")
    if update_mode != "none":
        print(f"  Final lag_scale = {model.lag_scale.item():.6f}")

    # Per-day R² stats
    day_r2s = np.array(list(day_results.values()))
    print(f"  Per-day R²: mean={day_r2s.mean():.6f}, std={day_r2s.std():.6f}, "
          f"min={day_r2s.min():.6f}, max={day_r2s.max():.6f}")

    return {
        "val_r2": float(total_r2),
        "n_updates": n_updates,
        "time_s": elapsed,
        "day_r2_mean": float(day_r2s.mean()),
        "day_r2_std": float(day_r2s.std()),
        "day_r2_min": float(day_r2s.min()),
        "day_r2_max": float(day_r2s.max()),
        "final_lag_scale": float(model.lag_scale.item()),
        "day_results": {int(k): float(v) for k, v in day_results.items()},
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # ─── Load normalization stats ─────────────────────────────────────────
    section("Loading normalization stats + model")
    stats = np.load(MODEL_DIR / "norm_stats.npz", allow_pickle=True)
    feat_mean = stats["mean"]
    feat_std = stats["std"]
    print(f"  Feature stats loaded: {len(feat_mean)} features")

    lag_init = float(0.90 * feat_std[LAG_R6_IDX])
    print(f"  lag_init_scale = {lag_init:.4f}")

    # Load trained GRU
    model_state = torch.load(MODEL_DIR / "gru_lag_residual.pt", map_location="cpu")
    print(f"  GRU state loaded ({len(model_state)} tensors)")

    # ─── Load validation data ─────────────────────────────────────────────
    section("Loading validation data for online simulation")
    X_raw, y, w, date_ids, time_ids, symbol_ids = load_online_data(
        VAL_LO, VAL_HI, overlap_start=VAL_OVERLAP_START)

    # Save un-normalized lag1_r6 (= true responder_6 from previous step)
    lag_r6_raw = X_raw[:, LAG_R6_IDX].copy()

    # Normalize features (in-place)
    X_norm = (X_raw - feat_mean) / feat_std
    del X_raw; gc.collect()

    # Build time step index
    time_steps = build_time_step_index(date_ids, time_ids)
    n_days = len(set(date_ids))
    print(f"  {len(time_steps):,} time steps across {n_days} days")
    print(f"  ~{len(time_steps)/n_days:.0f} steps/day, "
          f"~{len(y)/len(time_steps):.1f} symbols/step")

    # ─── Experiment 1: Frozen Baseline ────────────────────────────────────
    results = {}

    res = run_online_experiment(
        model_state, X_norm, y, w, date_ids, time_ids, symbol_ids,
        time_steps, lag_r6_raw, feat_mean, feat_std,
        update_mode="none",
        label="Exp 1: Frozen GRU (step-by-step baseline)")
    results["frozen_baseline"] = res

    # ─── Experiment 2: Head-only SGD, lr sweep ────────────────────────────
    for lr_val in [1e-5, 1e-4, 1e-3]:
        key = f"head_lr{lr_val:.0e}"
        res = run_online_experiment(
            model_state, X_norm, y, w, date_ids, time_ids, symbol_ids,
            time_steps, lag_r6_raw, feat_mean, feat_std,
            update_mode="head", lr=lr_val,
            label=f"Exp 2: Head-only SGD, lr={lr_val:.0e}")
        results[key] = res

    # ─── Experiment 3: All-layers SGD (best lr from head sweep) ───────────
    # Pick best head lr
    head_results = {k: v for k, v in results.items() if k.startswith("head_")}
    best_head_key = max(head_results, key=lambda k: head_results[k]["val_r2"])
    best_head_lr = float(best_head_key.split("lr")[1])
    print(f"\n  Best head lr: {best_head_lr:.0e} "
          f"(R²={head_results[best_head_key]['val_r2']:.6f})")

    for lr_val in [best_head_lr * 0.1, best_head_lr, best_head_lr * 10]:
        if lr_val > 0.01:
            continue
        key = f"all_lr{lr_val:.0e}"
        if key in results:
            continue
        res = run_online_experiment(
            model_state, X_norm, y, w, date_ids, time_ids, symbol_ids,
            time_steps, lag_r6_raw, feat_mean, feat_std,
            update_mode="all", lr=lr_val,
            label=f"Exp 3: All-layers SGD, lr={lr_val:.0e}")
        results[key] = res

    # ─── Experiment 4: Day boundary reset ─────────────────────────────────
    # Use best update config so far
    best_key = max(results, key=lambda k: results[k]["val_r2"])
    best_cfg = results[best_key]
    print(f"\n  Best so far: {best_key} (R²={best_cfg['val_r2']:.6f})")

    # Determine update_mode and lr from best key
    if "head" in best_key:
        reset_mode = "head"
        reset_lr = float(best_key.split("lr")[1])
    elif "all" in best_key:
        reset_mode = "all"
        reset_lr = float(best_key.split("lr")[1])
    else:
        reset_mode = "none"
        reset_lr = 1e-4

    if reset_mode != "none":
        res = run_online_experiment(
            model_state, X_norm, y, w, date_ids, time_ids, symbol_ids,
            time_steps, lag_r6_raw, feat_mean, feat_std,
            update_mode=reset_mode, lr=reset_lr,
            reset_weights_daily=True,
            label=f"Exp 4: {reset_mode} SGD lr={reset_lr:.0e} + daily weight reset")
        results["daily_reset"] = res

    # ─── Experiment 5: Lag EMA (no gradient) ──────────────────────────────
    for alpha in [0.01, 0.05, 0.1]:
        key = f"lag_ema_{alpha}"
        res = run_online_experiment(
            model_state, X_norm, y, w, date_ids, time_ids, symbol_ids,
            time_steps, lag_r6_raw, feat_mean, feat_std,
            update_mode="none", lag_ema_alpha=alpha,
            label=f"Exp 5: Lag EMA tracking (alpha={alpha})")
        results[key] = res

    # ─── Experiment 6: Combined best ──────────────────────────────────────
    # Best gradient config + lag EMA
    best_grad_key = max(
        [k for k in results if k.startswith(("head_", "all_"))],
        key=lambda k: results[k]["val_r2"], default=None)
    best_ema_key = max(
        [k for k in results if k.startswith("lag_ema_")],
        key=lambda k: results[k]["val_r2"], default=None)

    if best_grad_key and best_ema_key:
        if "head" in best_grad_key:
            combo_mode = "head"
        else:
            combo_mode = "all"
        combo_lr = float(best_grad_key.split("lr")[1])
        combo_alpha = float(best_ema_key.split("_")[-1])

        res = run_online_experiment(
            model_state, X_norm, y, w, date_ids, time_ids, symbol_ids,
            time_steps, lag_r6_raw, feat_mean, feat_std,
            update_mode=combo_mode, lr=combo_lr,
            lag_ema_alpha=combo_alpha,
            label=f"Exp 6: Combined ({combo_mode} lr={combo_lr:.0e} + EMA α={combo_alpha})")
        results["combined"] = res

    # ─── Summary ──────────────────────────────────────────────────────────
    section("ONLINE LEARNING RESULTS")
    print(f"\n  {'Experiment':<45} {'Val R²':>10} {'Δ vs frozen':>12} {'Updates':>8} {'Time':>6}")
    print("  " + "-" * 83)

    frozen_r2 = results["frozen_baseline"]["val_r2"]
    for key in sorted(results.keys(), key=lambda k: results[k]["val_r2"], reverse=True):
        r = results[key]
        delta = r["val_r2"] - frozen_r2
        print(f"  {key:<45} {r['val_r2']:10.6f} {delta:+12.6f} "
              f"{r['n_updates']:8d} {r['time_s']:5.0f}s")

    best_key = max(results, key=lambda k: results[k]["val_r2"])
    print(f"\n  Best: {best_key} (R²={results[best_key]['val_r2']:.6f})")

    # Save results
    # Strip day_results for compact JSON
    compact_results = {}
    for k, v in results.items():
        compact_results[k] = {kk: vv for kk, vv in v.items() if kk != "day_results"}
    with open(REPORT_DIR / "results.json", "w") as f:
        json.dump(compact_results, f, indent=2)
    print(f"\n  Results saved to {REPORT_DIR / 'results.json'}")
