"""Phase 6e: Hyperparameter Grid Search for GRU Architecture.

Motivation: Only one GRU config was tested in Phase 5 (2 layers, hidden=128,
chunk=64, dropout=0.1). Systematic search may find better architectures.

Search space:
  - Depth: 1, 2, 3 layers
  - Hidden dim: 64, 128, 256
  - Dropout: 0.0, 0.1, 0.2
  - Chunk size: 32, 64, 128 (affects sequence batching)
  - Learning rate: 5e-4, 1e-3, 2e-3

Total configs: 3 × 3 × 3 × 2 × 3 = 162 (too many for full search)
Strategy: Phase 1 coarse search (27 configs, 3 epochs each),
          Phase 2 fine search (top 5 configs, full training)

Usage:
    python hyperparameter_search.py              # full search
    python hyperparameter_search.py --coarse-only  # just phase 1
"""

import argparse
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
    TRAIN_LO, TRAIN_HI, VAL_LO, VAL_HI, VAL_OVERLAP_START,
    DEVICE, GRUModel,
    load_pass, standardize, predict_gru,
    find_sequence_groups, create_chunks, ChunkedSeqDataset, collate_sequences,
    WEIGHT_DECAY, EPOCHS, PATIENCE,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "hyperparameter_search"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Search space
GRID = {
    "n_layers": [1, 2, 3],
    "hidden_dim": [64, 128, 256],
    "dropout": [0.0, 0.1, 0.2],
    "chunk_size": [32, 64],
    "lr": [5e-4, 1e-3, 2e-3],
}

results = {}


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)


def train_config(cfg, X_tr, y_tr, w_tr, X_va, y_va, w_va,
                 date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                 lag_init, max_epochs=None, label=""):
    """Train one config. Returns val R² and training time."""
    torch.manual_seed(42)

    model = GRUModel(
        input_dim=len(ALL_FEATURES),
        lag_idx=LAG_R6_IDX,
        lag_init_scale=lag_init,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    groups_tr = find_sequence_groups(sym_ids_tr, date_ids_tr)
    chunks_tr = create_chunks(groups_tr, chunk_size=cfg["chunk_size"])

    dataset = ChunkedSeqDataset(X_tr, y_tr, w_tr, chunks_tr)
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    epochs = max_epochs or EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_r2, best_epoch = -np.inf, 0
    no_improve = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
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
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:  # Reduced patience for speed
                break

    train_time = time.time() - t0

    label_str = (f"  {label:<55} R²={val_r2:.6f} "
                 f"(best ep {best_epoch}, {train_time:.0f}s, {n_params:,} params)")
    print(label_str)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "val_r2": float(val_r2),
        "best_epoch": best_epoch,
        "train_time": train_time,
        "n_params": n_params,
        "config": dict(cfg),
    }


def cfg_label(cfg):
    return (f"L{cfg['n_layers']} H{cfg['hidden_dim']} D{cfg['dropout']} "
            f"C{cfg['chunk_size']} LR{cfg['lr']:.0e}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-only", action="store_true", help="Only run coarse search")
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

    # ─── Phase 1: Coarse Search (3 epochs each) ───────────────────────────
    section("Phase 1: Coarse Search (3 epochs per config)")

    configs = []
    for nl in GRID["n_layers"]:
        for hd in GRID["hidden_dim"]:
            for do in GRID["dropout"]:
                for cs in GRID["chunk_size"]:
                    for lr in GRID["lr"]:
                        configs.append({
                            "n_layers": nl, "hidden_dim": hd,
                            "dropout": do, "chunk_size": cs, "lr": lr,
                        })

    print(f"  Total configs: {len(configs)}")
    print(f"  {'Config':<55} {'Val R²':>8} {'Time':>8}")
    print("  " + "-" * 73)

    coarse_results = []
    t_start = time.time()
    for i, cfg in enumerate(configs):
        label = cfg_label(cfg)
        res = train_config(cfg, X_tr, y_tr, w_tr, X_va, y_va, w_va,
                           date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                           lag_init, max_epochs=3, label=label)
        coarse_results.append((cfg, res))
        results[f"coarse_{i}"] = {**res, "label": label}

        # Save progress every 10 configs
        if (i + 1) % 10 == 0:
            with open(REPORT_DIR / "results.json", "w") as f:
                json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n  Coarse search completed in {elapsed:.0f}s ({elapsed/len(configs):.1f}s/config)")

    # Sort by R²
    coarse_results.sort(key=lambda x: x[1]["val_r2"], reverse=True)
    print(f"\n  Top 10 coarse configs:")
    for i, (cfg, res) in enumerate(coarse_results[:10]):
        print(f"    {i+1}. {cfg_label(cfg)}  R²={res['val_r2']:.6f}  "
              f"({res['train_time']:.0f}s, {res['n_params']:,} params)")

    results["coarse_search_summary"] = {
        "total_configs": len(configs),
        "epochs_per_config": 3,
        "total_time": elapsed,
        "top_5": [
            {"config": cfg_label(cfg), "val_r2": res["val_r2"],
             "train_time": res["train_time"], "n_params": res["n_params"]}
            for cfg, res in coarse_results[:5]
        ],
    }

    if args.coarse_only:
        print("\n  Skipping fine search (--coarse-only flag set)")
        with open(REPORT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(0)

    # ─── Phase 2: Fine Search (full training, top 5) ──────────────────────
    section("Phase 2: Fine Search (full training, top 5 coarse configs)")

    top5 = coarse_results[:5]
    fine_results = []

    for i, (cfg, coarse_res) in enumerate(top5):
        label = f"Fine #{i+1}: {cfg_label(cfg)}"
        res = train_config(cfg, X_tr, y_tr, w_tr, X_va, y_va, w_va,
                           date_ids_tr, sym_ids_tr, date_ids_va, sym_ids_va,
                           lag_init, max_epochs=EPOCHS, label=label)
        fine_results.append((cfg, res))
        results[f"fine_{i}"] = {**res, "label": label, "coarse_r2": coarse_res["val_r2"]}
        torch.cuda.empty_cache(); gc.collect()

    # ─── Summary ───────────────────────────────────────────────────────────
    section("HYPERPARAMETER SEARCH RESULTS")

    print(f"\n  {'Config':<55} {'Coarse R²':>10} {'Fine R²':>10} {'Δ':>8}")
    print("  " + "-" * 85)

    baseline_r2 = 0.882852  # Phase 5 GRU
    print(f"  {'Phase 5 baseline (L2 H128 D0.1 C64 LR1e-3)':<53} {baseline_r2:10.6f}")
    print("  " + "-" * 85)

    for cfg, res in sorted(fine_results, key=lambda x: x[1]["val_r2"], reverse=True):
        delta = res["val_r2"] - baseline_r2
        print(f"  {cfg_label(cfg):<53} {res.get('coarse_r2', 0):10.6f} "
              f"{res['val_r2']:10.6f} {delta:+8.6f}")

    best_cfg, best_res = fine_results[0]
    if best_res["val_r2"] > baseline_r2 + 0.001:
        print(f"\n  ✓ Found better config: {cfg_label(best_cfg)} "
              f"(R²={best_res['val_r2']:.6f}, +{best_res['val_r2']-baseline_r2:+.6f})")
    else:
        print(f"\n  × No significant improvement over Phase 5 baseline.")
        print(f"  The default config (L2 H128 D0.1) is near-optimal.")

    # Save best config
    results["best_config"] = {
        "config": cfg_label(best_cfg),
        "val_r2": best_res["val_r2"],
        "train_r2": best_res.get("train_r2", 0),
        "n_params": best_res["n_params"],
        "train_time": best_res["train_time"],
        "improvement_over_baseline": best_res["val_r2"] - baseline_r2,
    }

    rp = REPORT_DIR / "results.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {rp}")
