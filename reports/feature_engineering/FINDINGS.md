# Phase 4: Feature Engineering — Findings

## Setup

- **Train**: dates 900-1188 (9,998,472 rows)
- **Val**: dates 1189-1443 (9,386,696 rows, overlap from 1174 for temporal continuity)
- **Target**: `responder_6`
- **Metric**: Weighted R²
- **Features engineered**: 89 total (54 base + 5 lag + 20 market + 10 rolling)

---

## Results Table

| Model                                        | Val R²     | Train R²   | Overfit Gap | Train Time | Inference  |
|----------------------------------------------|------------|------------|-------------|------------|------------|
| Phase 3: LightGBM (54 features, no lag)      | 0.010782   | 0.044423   | 0.034       | 121s       | 0.137ms    |
| Naive: predict lag_1 × 0.90                  | **0.811445** | —        | —           | —          | —          |
| LightGBM (54 sel + 5 lag)                    | **0.856097** | 0.868760 | 0.013       | 322s       | 0.111ms    |
| LightGBM (89 = sel + lag + market + rolling) | 0.855815   | 0.867539   | 0.012       | 370s       | 0.176ms    |

---

## Key Findings

### 1. Lag features are everything — they transform the problem

Adding 5 lag responder features (lag1_r6, r3, r7, r4, r0) to the 54 selected features
raises Val R² from **0.0108 → 0.856** — an 80× improvement.

This is not surprising given Phase 2 EDA (lag1_responder_6 has 0.90 correlation with
responder_6), but the magnitude is still striking. The model is essentially solving a
different problem when lags are available.

Feature importance breakdown (89-feature model):
- `lag1_r6` alone: **73.0% of gain**
- `lag1_r3`: **20.0% of gain**
- All other features combined: 7%

**Interpretation**: The signal is almost entirely about predicting that a responder
will continue near its most recent value (adjusted for mean-reversion). Features provide
incremental alpha on top of that autocorrelation.

### 2. LightGBM with lags (R²=0.856) beats the naive lag baseline (R²=0.811)

The feature-augmented model improves by **+0.045 R²** over the naive "predict lag_1 × 0.90"
baseline. This is meaningful alpha — it comes from:
- Learning the optimal lag-shrinkage coefficient (0.90 is approximate, model learns better)
- Nonlinear interactions between lag values (e.g., lag1_r3 interacts with lag1_r6)
- The remaining 54 features and market features providing incremental signal

### 3. Market and rolling features add negligible value

| Feature set          | Val R²     | Δ vs lag-only |
|----------------------|------------|----------------|
| 54 sel + 5 lag (59)  | 0.856097   | baseline       |
| 89 (+ market + roll) | 0.855815   | **−0.000282**  |

Market features (cross-symbol mean/std per time slot) and rolling features (10-step
rolling mean per symbol) add **no measurable improvement** over lag-only.

This makes sense in retrospect:
- `lag1_r6` (73% of gain) already captures the dominant autocorrelation signal
- `lag1_r3` (20% of gain) captures the cross-responder correlation
- Market and rolling features are second-order effects, drowned out by lag signal strength
- Only `feature_59` and a few lag features contribute meaningfully beyond top 2 lags

**Decision**: Use the 59-feature set (54 selected + 5 lag) for subsequent phases.
Market and rolling features are dropped — they don't help and add inference complexity.

### 4. Overfitting improves dramatically with lag features

| Model                  | Overfit Gap (train R² - val R²) |
|------------------------|---------------------------------|
| LightGBM, no lag (Ph3) | 0.034                           |
| LightGBM + lag         | 0.013                           |
| LightGBM + 89 features | 0.012                           |

When lags are included, the model spends most of its capacity on the dominant signal
(lag autocorrelation), leaving less room for overfitting on noise features. The overfit
gap dropped by **3×**.

### 5. Feature importance is now highly concentrated

Top 5 features dominate with 95%+ of total gain:
1. `lag1_r6`: 73%
2. `lag1_r3`: 20%
3. `feature_59`: 3.1%
4. `lag1_r0`: 1.1%
5. `lag1_r7`: 0.8%

The 54 original features are nearly irrelevant given the lags. However, the small
incremental alpha from features (+0.045 R² over naive lag) is real and worth preserving.

---

## Implications for Next Phases

### Phase 5: Neural Network

The NN should:
1. **Take lag1_r6 and lag1_r3 as primary inputs** — these are the dominant signals
2. **Architecture suggestion**: Predict = f(lag_features) + g(other_features), where f is
   a learned near-identity transformation and g provides incremental alpha
3. **Or equivalently**: Use lag1_r6 as a residual connection:
   `prediction = lag_scale × lag1_r6 + small_correction_from_features`
4. **Regularization**: Overfit gap is now 0.013 — still room for improvement. NN will need
   dropout, weight decay, and early stopping.
5. **Input dimension**: 59 features (54 + 5 lag) — or potentially just the 5 lag features
   plus the most important raw features (top 10-20 by gain from Phase 3)

### Phase 6: Online Learning

- The lag features are available at inference time (from lags.parquet)
- Online learning should focus on adapting the **lag_scale coefficient** (currently ~0.90)
- The scale may vary by market regime, symbol, and recent volatility
- A simple online linear regression on lag1_r6 alone could capture this

---

## Final Feature Set for Phase 5

**59 features:**
```python
LAG_FEATURES = ['lag1_r6', 'lag1_r3', 'lag1_r7', 'lag1_r4', 'lag1_r0']
SELECTED_54 = [...]  # from Phase 3 feature selection
ALL_FEATURES = SELECTED_54 + LAG_FEATURES  # 59 total
```

Or for a more compact set (top contributors only):
```python
TOP_FEATURES = ['lag1_r6', 'lag1_r3', 'feature_59', 'lag1_r0', 'lag1_r7',
                'feature_06', 'feature_60', 'feature_13', 'feature_52', 'feature_49']
```

---

## Files

- `reports/feature_engineering/results.json` — numeric results
- `src/feature_engineering.py` — full Phase 4 pipeline
