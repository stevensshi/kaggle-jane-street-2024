# Phase 5: Neural Network — Findings

## Setup

- **Train**: dates 900-1188 (9,998,472 rows)
- **Val**: dates 1189-1443 (9,386,696 rows)
- **Features**: 59 (54 selected + 5 lag responders)
- **Target**: `responder_6`
- **Metric**: Weighted R²
- **GPU**: RTX 4060 (8.6 GB VRAM), CUDA, PyTorch 2.11

---

## Results Table

| Model                              | Val R²     | Train R²   | Gap    | Time  | Infer   | Params  |
|------------------------------------|------------|------------|--------|-------|---------|---------|
| Naive: predict lag_1 × 0.90       | 0.811445   | —          | —      | —     | —       | —       |
| LightGBM (59 features, Phase 4)   | 0.856097   | 0.868760   | 0.013  | 322s  | 0.111ms | —       |
| MLP baseline                       | 0.858997   | 0.863768   | 0.005  | 1224s | 0.466ms | 115,457 |
| MLP + lag residual                 | 0.861152   | 0.865554   | 0.004  | 1437s | 3.550ms | 115,459 |
| Multi-task MLP (r6+r3)            | 0.861668   | 0.865894   | 0.004  | 2518s | 0.607ms | 148,228 |
| **GRU + lag residual**            | **0.882851** | 0.886656 | 0.004  | 304s  | 0.372ms | 179,971 |

---

## Key Findings

### 1. GRU dramatically outperforms all other models (R²=0.883)

The GRU with lag residual achieves **R²=0.8829**, beating:
- LightGBM by **+0.0268** (a massive improvement)
- Best MLP (multi-task) by **+0.0212**
- Naive lag baseline by **+0.0714**

This proves **temporal sequence modeling within each (symbol, day) matters**.
The GRU captures intraday patterns that pointwise models (MLP, LightGBM) cannot.

### 2. GRU is also the fastest to train

Despite being the most complex model, GRU trained in only **304 seconds** (5 min)
vs 1200-2500s for MLPs. This is because:
- Chunked sequences (size 64) with efficient batching (128 sequences/batch)
- Packed sequences handle variable lengths without wasted compute
- Fewer effective parameter updates needed (165K chunks vs 10M individual rows)

### 3. MLP variants beat LightGBM but by a smaller margin

| MLP Variant      | Val R²   | vs LightGBM |
|------------------|----------|-------------|
| Baseline         | 0.8590   | +0.0029     |
| + Lag residual   | 0.8612   | +0.0051     |
| + Multi-task     | 0.8617   | +0.0056     |

All three MLP variants surpass LightGBM. The lag residual adds +0.002 R² over
baseline MLP, and multi-task adds another +0.0005. These are modest gains
compared to the GRU's +0.027.

### 4. Multi-task learning provides small but consistent improvement

Adding responder_3 as an auxiliary target (aux_weight=0.3) improved the MLP from
0.8612 → 0.8617 (+0.0005). While small, this suggests the shared encoder learns
slightly better representations when trained on correlated tasks.

### 5. All NN models have lower overfitting than LightGBM

| Model          | Train-Val Gap |
|----------------|---------------|
| LightGBM       | 0.0127        |
| MLP baseline   | 0.0048        |
| MLP+lag res    | 0.0044        |
| Multi-task MLP | 0.0042        |
| GRU+lag res    | 0.0038        |

NN models overfit 3× less than LightGBM. Dropout, weight decay, and
early stopping are effective regularizers. The GRU has the tightest gap.

### 6. Inference speed is well within 16ms budget

All models are under 4ms per row/step — well within the 16ms competition budget.
The GRU at 0.372ms/step is particularly efficient.

---

## Architecture Details

### Lag Residual Connection

All models (except MLP baseline) use:
```
prediction = lag_scale × lag1_r6 + lag_bias + correction(features)
```

- `lag_scale` initialized at `0.90 × std(lag1_r6)` to account for feature standardization
- `lag_bias` initialized at 0
- Correction network's last layer initialized at zeros (starts as pure lag prediction)
- The model learns to add small corrections on top of the dominant lag signal

### GRU Architecture

- Input: 59 features per time step
- GRU: 2 layers, hidden_dim=128, dropout=0.1
- Head: Linear(128, 64) → SiLU → Linear(64, 1) (initialized near zero)
- Sequences: per (symbol_id, date_id), sorted by time_id
- Training chunks: size 64 (from ~900-step sequences), min length 4
- 10,329 groups → 165,264 chunks

### Multi-Task MLP

- Shared encoder: 59 → 256 → 256 (with BatchNorm, SiLU, Dropout)
- Primary head (r6): 256 → 128 → 1 (with lag residual)
- Auxiliary head (r3): 256 → 128 → 1
- Loss: weighted_MSE(r6) + 0.3 × weighted_MSE(r3)

---

## Implications for Phase 6 (Online Learning)

1. **GRU is the architecture to take forward** — it's the best model by a large margin
2. **Online learning should adapt the GRU hidden states** — they capture intraday patterns
3. **The lag_scale parameter is a natural target for online adaptation** — it may vary by regime
4. **Fast inference (0.37ms)** leaves ample budget for online update computation

---

## Files

- `reports/neural_network/results.json` — numeric results
- `models/mlp_baseline.pt` — MLP baseline weights
- `models/mlp_lag_residual.pt` — MLP + lag residual weights
- `models/multitask_mlp.pt` — Multi-task MLP weights
- `models/gru_lag_residual.pt` — GRU + lag residual weights (best model)
- `models/norm_stats.npz` — feature normalization (mean, std)
- `src/neural_network.py` — full Phase 5 pipeline
