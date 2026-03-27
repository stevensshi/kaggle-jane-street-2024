# Phase 3: Baseline Models — Findings

## Setup

- **Train**: dates 900-1188 (9,998,472 rows, ~289 trading days)
- **Val**: dates 1189-1443 (9,386,696 rows, ~255 trading days)
- **Target**: `responder_6` (the primary scoring target)
- **Metric**: Weighted R² (sample weights = `weight` column, mean ~2.0)
- **Features**: 79 anonymized features (`feature_00` through `feature_78`)

---

## Results Table

| Model                    | Val R²     | Train Time | Inference  |
|--------------------------|------------|------------|------------|
| Predict Zero             | 0.000000   | —          | —          |
| Predict lag_1 (raw)      | 0.801605   | —          | —          |
| Predict lag_1 (scaled)   | **0.811445** | —        | —          |
| Ridge (top 5 features)   | 0.002406   | 0.8s       | —          |
| Ridge (all 79 features)  | 0.004260   | 6.6s       | 0.045ms    |
| Ridge (54 selected)      | 0.004385   | 5.6s       | 0.045ms    |
| LightGBM (all 79)        | 0.010825   | 140s       | 0.137ms    |
| LightGBM (54 selected)   | 0.010782   | 121s       | 0.137ms    |
| XGBoost (54 selected)    | 0.010605   | 108s       | 0.148ms    |

---

## Key Findings

### 1. Lag_1 is the dominant signal — by far

The simplest possible model — "predict that responder_6 will equal its value from the
previous time step" — achieves R² = 0.80. With optimal shrinkage (scale = 0.90), it
reaches R² = 0.81.

Meanwhile, the best feature-only model (LightGBM on all 79 features) achieves R² = 0.011.
That's **~75x weaker** than the lag signal.

**Interpretation**: This is a highly autocorrelated return signal. The market microstructure
here has strong mean-reversion or persistence at short horizons. The 0.90 shrinkage factor
suggests a slight mean-reversion bias — the optimal prediction is 90% of the previous
value, not 100%.

**Implication for modeling**: The lag feature is not available as an input feature (it comes
from the competition's `lags.parquet` at inference time). Any competitive solution MUST
incorporate lag_1 as the primary signal. The features provide incremental alpha on top.

### 2. Nonlinear interactions matter (trees >> linear)

| Model Type | R² on 54 features |
|------------|-------------------|
| Ridge      | 0.0044            |
| XGBoost    | 0.0106            |
| LightGBM   | 0.0108            |

Tree-based models deliver **~2.5x** higher R² than Ridge on the same features. This
confirms the features contain nonlinear interactions (splits, thresholds, conditional
effects) that linear models cannot capture.

**Implication**: Neural networks, which can also capture nonlinearities, are a natural
next step. The incremental R² from features (0.011) is small in absolute terms but could
be meaningful when combined with the lag signal in production.

### 3. Feature importance is concentrated but stable

**Concentration**: The top 18 features capture 50% of LightGBM's total gain. The top
54 capture 90%. The bottom 25 features contribute almost nothing.

**Top 6 features by gain**:
1. `feature_06` (5.4%) — also the strongest linear correlate from EDA
2. `feature_61` (4.2%)
3. `feature_30` (3.4%)
4. `feature_36` (3.3%)
5. `feature_07` (3.2%)
6. `feature_04` (3.0%)

**Stability**: Spearman rank correlation of feature importances between early (dates
900-1044) and late (dates 1045-1188) training halves is **0.904** — very high. Out of
30 top features in each half, **28 overlap**. The signal structure is not drifting
significantly over this ~144-day window.

**Noise features**: `feature_63` (6 splits, lowest gain/split), `feature_55`, `feature_41`,
`feature_00` are the noisiest — used by the trees but contributing almost nothing per
split. Safe to drop.

**Feature selection result**: Dropping from 79 to 54 features loses essentially zero
performance (R² 0.01082 vs 0.01083). This is a free complexity reduction.

### 4. LightGBM overfits significantly

- Train R²: 0.0444
- Val R²: 0.0108
- **Overfit gap: 0.034** (train is 4x higher)

Even with early stopping at 111 rounds (from 2000), `min_child_samples=100`,
`subsample=0.8`, and L1/L2 regularization, the model still overfits substantially.

This suggests the signal is very noisy (the target has std ~0.87 with a near-zero mean).
The R² of ~0.01 means we're explaining about 1% of variance — the remaining 99% is noise.
In such a regime, any model with enough capacity will memorize noise.

**Implication**: For the neural network phase, aggressive regularization (dropout, weight
decay, early stopping) will be critical. Ensemble methods that average over multiple
models can also help.

### 5. All models are fast enough for inference

The competition requires predictions within a time budget. All models produce predictions
in well under 1ms per row:

| Model    | Inference/row |
|----------|---------------|
| Ridge    | 0.045ms       |
| LightGBM | 0.137ms       |
| XGBoost  | 0.148ms       |

Even a 10-model ensemble would still be under 2ms/row, far below typical competition
budgets of 5-16ms/row. Inference speed is not a binding constraint for model choice.

### 6. LightGBM and XGBoost are nearly identical

LightGBM: R² = 0.01078, XGBoost: R² = 0.01061. The difference is within noise. Both
early-stopped at ~110-140 rounds. Training times are comparable (121s vs 108s).

**Decision**: Use LightGBM as the tree baseline going forward (marginally better R²,
better API for feature importance, wider use in Kaggle winning solutions).

---

## Implications for Next Phases

1. **Phase 4 (GRU/NN)**: The NN must take lag_1 as the primary input. Features provide
   incremental signal (~1% of variance). Architecture should:
   - Accept lag_1 + features as inputs
   - Use the lag heavily (perhaps as a residual connection: predict = lag_1 * scale + NN(features))
   - Apply strong regularization given the 4x overfit gap seen in LightGBM

2. **Phase 5 (Online Learning)**: With lag_1 correlation of 0.80+, online recalibration
   of the lag coefficient alone could be valuable. The feature-based component might
   benefit from periodic refitting or exponential weighting of recent data.

3. **Phase 6 (Ensemble)**: Combine lag_1 baseline + LightGBM features + NN for
   diversification. Even small uncorrelated alpha from different model families adds up
   when the base signal (lag) is so strong.

---

## Files

- `reports/baselines/results.json` — all numeric results
- `src/baselines.py` — full Phase 3 pipeline (memory-optimized)
- `src/data.py` — data loading utilities
- `src/evaluate.py` — weighted R² metric
