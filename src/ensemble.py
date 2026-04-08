"""Phase 7: Ensemble & Final Holdout Evaluation.

1. Multi-seed GRU ensemble (5 seeds)
2. Cross-architecture ensemble (GRU + MLP)
3. Holdout test evaluation (dates 1444-1698) — ONE TIME ONLY

Base: GRU lag-residual (Val R² = 0.883), MLP lag-residual (Val R² = 0.861)
"""

import argparse
import gc
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import weighted_r2
from neural_network import (
    TRAIN_PATH, ALL_FEATURES, LAG_R6_IDX, TARGET_COL, AUX_TARGET,
    SELECTED_54, LAG_RESPONDERS,
    TRAIN_LO, TRAIN_HI, VAL_LO, VAL_HI, VAL_OVERLAP_START,
    DEVICE, GRUModel, MLPLagResidual,
    load_pass, standardize, predict_gru, predict_mlp,
    find_sequence_groups, create_chunks, ChunkedSeqDataset, collate_sequences,
    train_epoch_gru,
    BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, PATIENCE,
)
from torch.utils.data import DataLoader

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "ensemble"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_LO, HOLDOUT_HI = 1444, 1698
HOLDOUT_OVERLAP_START = 1429  # 15-day overlap for lag warmup

N_SEEDS = 5
GRU_SEEDS = [42, 123, 456, 789, 2024]


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)


def train_gru_seed(seed, X_tr, y_tr, w_tr, X_va, y_va, w_va,
                   date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                   lag_init, chunk_size=64, seq_batch_size=128):
    """Train one GRU with a specific seed. Returns state_dict and val R²."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                     lag_init_scale=lag_init).to(DEVICE)

    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=chunk_size)

    dataset = ChunkedSeqDataset(X_tr, y_tr, w_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=seq_batch_size, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0, pin_memory=True)

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

        if val_r2 > best_r2:
            best_r2, best_epoch = val_r2, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    train_time = time.time() - t0

    # Reload best state
    model.load_state_dict(best_state)
    model.to(DEVICE)

    # Train R²
    tr_pred = predict_gru(model, X_tr, date_ids_tr, sym_ids_tr)
    tr_r2 = weighted_r2(y_tr, tr_pred, w_tr)

    print(f"  Seed {seed}: Val R²={best_r2:.6f}, Train R²={tr_r2:.6f}, "
          f"gap={tr_r2-best_r2:.4f}, epoch={best_epoch}, time={train_time:.0f}s")

    del model; torch.cuda.empty_cache()
    return best_state, best_r2, tr_r2, best_epoch, train_time


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    results = {}

    # ─── Load Training + Validation Data ──────────────────────────────────

    section("Loading training data (dates 900-1188)")
    X_tr, y_tr, w_tr, _, date_ids_tr, sym_ids_tr = load_pass(TRAIN_LO, TRAIN_HI)

    section("Loading validation data (dates 1189-1443)")
    X_va, y_va, w_va, _, date_ids_va, sym_ids_va = load_pass(
        VAL_LO, VAL_HI, overlap_start=VAL_OVERLAP_START)

    section("Standardizing features")
    X_tr_norm, X_va_norm, feat_mean, feat_std = standardize(X_tr, X_va)
    lag_init = float(0.90 * feat_std[LAG_R6_IDX])
    print(f"  lag_init_scale = {lag_init:.4f}")

    # ═════════════════════════════════════════════════════════════════════
    # Phase 7.1: Multi-Seed GRU Ensemble
    # ═════════════════════════════════════════════════════════════════════

    section(f"Training {N_SEEDS}-seed GRU ensemble")

    seed_states = []
    seed_results = []

    for i, seed in enumerate(GRU_SEEDS):
        print(f"\n  --- Seed {i+1}/{N_SEEDS} (seed={seed}) ---")
        state, val_r2, tr_r2, best_ep, train_t = train_gru_seed(
            seed, X_tr_norm, y_tr, w_tr, X_va_norm, y_va, w_va,
            date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va, lag_init)
        seed_states.append(state)
        seed_results.append({
            "seed": seed, "val_r2": val_r2, "train_r2": tr_r2,
            "best_epoch": best_ep, "train_time": train_t,
        })
        # Save individual model
        torch.save(state, MODEL_DIR / f"gru_seed{seed}.pt")

    # Ensemble predictions on validation
    section("Evaluating multi-seed ensemble on validation")
    ensemble_preds_va = np.zeros(len(y_va), dtype=np.float32)

    for i, state in enumerate(seed_states):
        model = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                         lag_init_scale=lag_init).to(DEVICE)
        model.load_state_dict(state)
        preds = predict_gru(model, X_va_norm, date_ids_va, sym_ids_va)
        ensemble_preds_va += preds
        del model; torch.cuda.empty_cache()

    ensemble_preds_va /= N_SEEDS
    ensemble_val_r2 = weighted_r2(y_va, ensemble_preds_va, w_va)

    individual_r2s = [r["val_r2"] for r in seed_results]
    print(f"  Individual Val R²: {[f'{r:.6f}' for r in individual_r2s]}")
    print(f"  Mean individual:   {np.mean(individual_r2s):.6f}")
    print(f"  Ensemble Val R²:   {ensemble_val_r2:.6f}")
    print(f"  Ensemble gain:     {ensemble_val_r2 - np.mean(individual_r2s):+.6f}")

    results["multi_seed_ensemble"] = {
        "n_seeds": N_SEEDS,
        "individual_r2s": [float(r) for r in individual_r2s],
        "mean_individual_r2": float(np.mean(individual_r2s)),
        "ensemble_val_r2": float(ensemble_val_r2),
        "seeds": seed_results,
    }

    # ═════════════════════════════════════════════════════════════════════
    # Phase 7.2: Cross-Architecture Ensemble (GRU + MLP)
    # ═════════════════════════════════════════════════════════════════════

    section("Cross-architecture ensemble (GRU + MLP)")

    # Load best single GRU (seed 42, original)
    gru = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                   lag_init_scale=lag_init).to(DEVICE)
    gru.load_state_dict(torch.load(MODEL_DIR / "gru_lag_residual.pt", map_location="cpu"))
    gru_preds_va = predict_gru(gru, X_va_norm, date_ids_va, sym_ids_va)
    gru_r2 = weighted_r2(y_va, gru_preds_va, w_va)
    del gru; torch.cuda.empty_cache()

    # Load MLP
    mlp = MLPLagResidual(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                         lag_init_scale=lag_init).to(DEVICE)
    mlp.load_state_dict(torch.load(MODEL_DIR / "mlp_lag_residual.pt", map_location="cpu"))
    mlp_preds_va = predict_mlp(mlp, X_va_norm)
    mlp_r2 = weighted_r2(y_va, mlp_preds_va, w_va)
    del mlp; torch.cuda.empty_cache()

    print(f"  GRU single:  R² = {gru_r2:.6f}")
    print(f"  MLP single:  R² = {mlp_r2:.6f}")

    # Simple average
    avg_preds = (gru_preds_va + mlp_preds_va) / 2
    avg_r2 = weighted_r2(y_va, avg_preds, w_va)
    print(f"  GRU+MLP avg: R² = {avg_r2:.6f}")

    # Weighted blends
    best_blend_r2 = -np.inf
    best_alpha = 0.5
    for alpha in np.arange(0.5, 1.01, 0.05):
        blend = alpha * gru_preds_va + (1 - alpha) * mlp_preds_va
        r2 = weighted_r2(y_va, blend, w_va)
        if r2 > best_blend_r2:
            best_blend_r2 = r2
            best_alpha = alpha
        print(f"  GRU×{alpha:.2f} + MLP×{1-alpha:.2f}: R² = {r2:.6f}")

    # GRU ensemble + MLP
    for alpha in [0.8, 0.85, 0.9, 0.95]:
        blend = alpha * ensemble_preds_va + (1 - alpha) * mlp_preds_va
        r2 = weighted_r2(y_va, blend, w_va)
        print(f"  GRU-ens×{alpha:.2f} + MLP×{1-alpha:.2f}: R² = {r2:.6f}")

    results["cross_architecture"] = {
        "gru_r2": float(gru_r2),
        "mlp_r2": float(mlp_r2),
        "simple_avg_r2": float(avg_r2),
        "best_blend_alpha": float(best_alpha),
        "best_blend_r2": float(best_blend_r2),
    }

    # Free validation data
    del X_tr, X_tr_norm, gru_preds_va, mlp_preds_va, avg_preds, ensemble_preds_va
    gc.collect(); torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════════════════
    # Phase 7.3: Inference Timing
    # ═════════════════════════════════════════════════════════════════════

    section("Inference timing (single step)")
    model = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                     lag_init_scale=lag_init).to(DEVICE)
    model.load_state_dict(seed_states[0])
    model.eval()

    # Single step: (1, 1, features) with hidden state
    x1 = torch.randn(1, 1, len(ALL_FEATURES), device=DEVICE)
    h0 = torch.zeros(2, 1, 128, device=DEVICE)

    # Warmup
    for _ in range(100):
        _, _ = model.gru(x1, h0)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10000):
        out, h_new = model.gru(x1, h0)
        _ = model.head(out.squeeze(1))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    single_step_ms = (time.time() - t0) / 10000 * 1000

    # Batch of 39 symbols (one time step)
    x39 = torch.randn(39, 1, len(ALL_FEATURES), device=DEVICE)
    h39 = torch.zeros(2, 39, 128, device=DEVICE)

    for _ in range(100):
        _, _ = model.gru(x39, h39)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10000):
        out, h_new = model.gru(x39, h39)
        _ = model.head(out.squeeze(1))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    batch_step_ms = (time.time() - t0) / 10000 * 1000

    print(f"  Single symbol step: {single_step_ms:.3f}ms")
    print(f"  39-symbol batch step: {batch_step_ms:.3f}ms")
    print(f"  Well within 16ms budget: {'YES' if batch_step_ms < 16 else 'NO'}")

    results["inference_timing"] = {
        "single_step_ms": single_step_ms,
        "batch_39_step_ms": batch_step_ms,
    }

    del model; torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════════════════
    # Phase 7.4: HOLDOUT TEST EVALUATION (ONE TIME ONLY)
    # ═════════════════════════════════════════════════════════════════════

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-holdout", action="store_true",
                        help="Evaluate on holdout set. Hard-locked: aborts if already run.")
    args, _ = parser.parse_known_args()

    HOLDOUT_ARTIFACT = REPORT_DIR / "holdout_result.json"

    if not args.run_holdout:
        print("\n  [HOLDOUT SKIPPED] Pass --run-holdout to evaluate the holdout set.")
        print("  This flag is intentionally one-shot: once run, results are locked.")
        with open(REPORT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results (no holdout) saved to {REPORT_DIR / 'results.json'}")
        sys.exit(0)

    if HOLDOUT_ARTIFACT.exists():
        raise RuntimeError(
            f"\n  HOLDOUT ALREADY EVALUATED.\n"
            f"  Results locked at: {HOLDOUT_ARTIFACT}\n"
            f"  Delete this file manually only if you understand the consequences "
            f"(doing so invalidates the unbiased estimate)."
        )

    section("HOLDOUT TEST EVALUATION (dates 1444-1698)")
    print("  *** This is a ONE-TIME evaluation. Results are final. ***\n")

    # Load holdout data
    X_ho, y_ho, w_ho, _, date_ids_ho, sym_ids_ho = load_pass(
        HOLDOUT_LO, HOLDOUT_HI, overlap_start=HOLDOUT_OVERLAP_START)
    print(f"  Holdout: {len(y_ho):,} rows, dates {HOLDOUT_LO}-{HOLDOUT_HI}")

    # Normalize with TRAINING stats
    X_ho_norm = (X_ho - feat_mean) / feat_std
    del X_ho; gc.collect()

    # --- Single best GRU (original seed 42) ---
    gru = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                   lag_init_scale=lag_init).to(DEVICE)
    gru.load_state_dict(torch.load(MODEL_DIR / "gru_lag_residual.pt", map_location="cpu"))
    gru_preds_ho = predict_gru(gru, X_ho_norm, date_ids_ho, sym_ids_ho)
    gru_ho_r2 = weighted_r2(y_ho, gru_preds_ho, w_ho)
    del gru; torch.cuda.empty_cache()
    print(f"  GRU single (seed 42):  Holdout R² = {gru_ho_r2:.6f}  "
          f"(Val R² was {gru_r2:.6f}, gap={gru_r2-gru_ho_r2:+.6f})")

    # --- Multi-seed ensemble ---
    ensemble_preds_ho = np.zeros(len(y_ho), dtype=np.float32)
    for state in seed_states:
        model = GRUModel(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                         lag_init_scale=lag_init).to(DEVICE)
        model.load_state_dict(state)
        ensemble_preds_ho += predict_gru(model, X_ho_norm, date_ids_ho, sym_ids_ho)
        del model; torch.cuda.empty_cache()
    ensemble_preds_ho /= N_SEEDS
    ensemble_ho_r2 = weighted_r2(y_ho, ensemble_preds_ho, w_ho)
    print(f"  GRU ensemble ({N_SEEDS} seeds): Holdout R² = {ensemble_ho_r2:.6f}  "
          f"(Val R² was {results['multi_seed_ensemble']['ensemble_val_r2']:.6f})")

    # --- MLP ---
    mlp = MLPLagResidual(input_dim=len(ALL_FEATURES), lag_idx=LAG_R6_IDX,
                         lag_init_scale=lag_init).to(DEVICE)
    mlp.load_state_dict(torch.load(MODEL_DIR / "mlp_lag_residual.pt", map_location="cpu"))
    mlp_preds_ho = predict_mlp(mlp, X_ho_norm)
    mlp_ho_r2 = weighted_r2(y_ho, mlp_preds_ho, w_ho)
    del mlp; torch.cuda.empty_cache()
    print(f"  MLP lag-residual:      Holdout R² = {mlp_ho_r2:.6f}  "
          f"(Val R² was {mlp_r2:.6f})")

    # --- Naive lag baseline ---
    lag_r6_ho_raw = X_ho_norm[:, LAG_R6_IDX] * feat_std[LAG_R6_IDX] + feat_mean[LAG_R6_IDX]
    naive_ho_r2 = weighted_r2(y_ho, 0.90 * lag_r6_ho_raw, w_ho)
    print(f"  Naive 0.90×lag:        Holdout R² = {naive_ho_r2:.6f}")

    # --- Best blend on holdout ---
    blend_ho = best_alpha * gru_preds_ho + (1 - best_alpha) * mlp_preds_ho
    blend_ho_r2 = weighted_r2(y_ho, blend_ho, w_ho)
    print(f"  GRU×{best_alpha:.2f}+MLP×{1-best_alpha:.2f}:  "
          f"Holdout R² = {blend_ho_r2:.6f}")

    results["holdout"] = {
        "n_rows": len(y_ho),
        "dates": f"{HOLDOUT_LO}-{HOLDOUT_HI}",
        "naive_lag_r2": float(naive_ho_r2),
        "mlp_r2": float(mlp_ho_r2),
        "gru_single_r2": float(gru_ho_r2),
        "gru_ensemble_r2": float(ensemble_ho_r2),
        "blend_r2": float(blend_ho_r2),
        "blend_alpha": float(best_alpha),
    }

    # ─── Final Summary ────────────────────────────────────────────────────
    section("FINAL RESULTS SUMMARY")

    print(f"\n  {'Model':<35} {'Val R²':>10} {'Holdout R²':>12} {'Gap':>8}")
    print("  " + "-" * 67)
    print(f"  {'Naive 0.90×lag':<35} {'0.8114':>10} {naive_ho_r2:12.6f}")
    print(f"  {'MLP lag-residual':<35} {mlp_r2:10.6f} {mlp_ho_r2:12.6f} "
          f"{mlp_r2-mlp_ho_r2:+8.6f}")
    print(f"  {'GRU single (seed 42)':<35} {gru_r2:10.6f} {gru_ho_r2:12.6f} "
          f"{gru_r2-gru_ho_r2:+8.6f}")
    print(f"  {f'GRU ensemble ({N_SEEDS} seeds)':<35} "
          f"{results['multi_seed_ensemble']['ensemble_val_r2']:10.6f} "
          f"{ensemble_ho_r2:12.6f} "
          f"{results['multi_seed_ensemble']['ensemble_val_r2']-ensemble_ho_r2:+8.6f}")
    print(f"  {'Best blend':<35} {best_blend_r2:10.6f} {blend_ho_r2:12.6f} "
          f"{best_blend_r2-blend_ho_r2:+8.6f}")

    print(f"\n  Inference: single={single_step_ms:.3f}ms, "
          f"batch39={batch_step_ms:.3f}ms (budget=16ms)")

    # Save full results
    with open(REPORT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {REPORT_DIR / 'results.json'}")

    # Lock holdout — persist metadata so re-running is detectable
    holdout_meta = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_dates": f"{HOLDOUT_LO}-{HOLDOUT_HI}",
        "n_rows": len(y_ho),
        "model_config": {
            "n_seeds": N_SEEDS,
            "seeds": GRU_SEEDS,
            "blend_alpha": float(best_alpha),
        },
        "scores": {
            "naive_lag_r2": float(naive_ho_r2),
            "mlp_r2": float(mlp_ho_r2),
            "gru_single_r2": float(gru_ho_r2),
            "gru_ensemble_r2": float(ensemble_ho_r2),
            "blend_r2": float(blend_ho_r2),
        },
    }
    with open(HOLDOUT_ARTIFACT, "w") as f:
        json.dump(holdout_meta, f, indent=2)
    print(f"  Holdout locked at: {HOLDOUT_ARTIFACT}")
