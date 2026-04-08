# Phase 7: Ensemble & Final Holdout Evaluation — Findings

## Setup

- **Training**: dates 900-1188 (10M rows)
- **Validation**: dates 1189-1443 (9.4M rows)
- **Holdout**: dates 1444-1698 (9.4M rows) — ONE-TIME evaluation
- **GPU**: RTX 4060, CUDA

---

## Phase 7.1: Multi-Seed GRU Ensemble

### Individual Seeds

| Seed | Val R²   | Train R² | Gap    | Epoch | Time |
|------|----------|----------|--------|-------|------|
| 42   | 0.882851 | 0.886656 | 0.0038 | 7     | 304s |
| 123  | 0.882564 | —        | —      | —     | —    |
| 456  | 0.882569 | 0.885882 | 0.0033 | 6     | 285s |
| 789  | 0.882540 | 0.886641 | 0.0041 | 7     | 312s |
| 2024 | 0.882319 | 0.885414 | 0.0031 | 6     | 281s |

### Ensemble

- **Mean individual Val R²**: 0.882569
- **5-seed ensemble Val R²**: 0.883888
- **Ensemble gain**: +0.001319

The 5-seed ensemble provides a modest but consistent improvement over any
single model. Individual models are remarkably stable across seeds (std < 0.0002).

---

## Phase 7.2: Cross-Architecture Ensemble

### GRU + MLP Blending (Validation)

| Blend                         | Val R²     |
|-------------------------------|------------|
| MLP single                    | 0.861152   |
| GRU single                    | 0.882851   |
| GRU 50% + MLP 50%            | 0.877668   |
| GRU 90% + MLP 10%            | 0.882721   |
| **GRU 100% (no MLP)**        | **0.882851** |
| GRU-ens 85% + MLP 15%        | 0.883262   |
| GRU-ens 90% + MLP 10%        | 0.883580   |
| **GRU-ens 95% + MLP 5%**     | **0.883789** |
| GRU-ens 100%                  | 0.883888   |

Adding MLP to the GRU ensemble hurts performance. The GRU dominates so heavily
that any MLP contribution dilutes the signal. **Best approach: pure GRU ensemble.**

---

## Phase 7.3: Inference Timing

| Setup               | Time     |
|---------------------|----------|
| Single symbol step  | 0.257ms  |
| 39-symbol batch step| 0.276ms  |
| Budget              | 16ms     |

The GRU is **58× faster** than the 16ms budget. Ample room for ensemble
averaging or even online update overhead if needed.

---

## Phase 7.4: Holdout Test Evaluation (FINAL)

| Model                    | Val R²     | Holdout R²   | Val→Holdout Gap |
|--------------------------|------------|--------------|-----------------|
| Naive 0.90×lag           | 0.8114     | 0.811466     | −0.000          |
| MLP lag-residual         | 0.861152   | 0.856541     | +0.004611       |
| **GRU single (seed 42)** | 0.882851  | **0.879970** | +0.002881       |
| **GRU ensemble (5 seeds)** | 0.883888 | **0.880850** | +0.003038      |

### Key Observations

1. **GRU ensemble wins on holdout**: R² = 0.8809, improving over single GRU by +0.0009.

2. **Small val→holdout gap**: All models show ~0.003-0.005 degradation from val to holdout.
   This is remarkably small, indicating the lag autocorrelation structure is stable
   across all historical dates (1189-1698).

3. **Naive lag baseline is perfectly stable**: R² = 0.8114 on both val and holdout,
   confirming the 0.90 lag autocorrelation holds throughout the historical data.

4. **Model rankings are preserved**: GRU > MLP > Naive on both val and holdout.

5. **Competition context**: Our holdout R² = 0.88 vs competition best score of 0.013.
   This confirms the user's insight: the lag autocorrelation of 0.90 is a historical
   artifact that does not persist on truly future market data. Our models are optimized
   for a signal that vanishes in production.

---

## Final Model Summary

| Property         | Value                        |
|------------------|------------------------------|
| Architecture     | GRU + lag residual (5 seeds) |
| Val R²           | 0.883888                     |
| Holdout R²       | 0.880850                     |
| Parameters       | 180K per seed                |
| Inference        | 0.276ms per batch step       |
| Features         | 59 (54 selected + 5 lag)     |

---

## Files

- `reports/ensemble/results.json` — full numeric results
- `models/gru_seed{42,123,456,789,2024}.pt` — individual seed weights
- `models/gru_lag_residual.pt` — original seed 42 weights
- `models/norm_stats.npz` — normalization stats
- `src/ensemble.py` — full ensemble + holdout pipeline
