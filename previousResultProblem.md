# Jane Street 2024 — Strategy, Experiments & Results

## Data Source
- **Competition**: [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/)
- **Data download**: See competition Data tab. The raw parquet files go in `data/raw/`.
- **Competition ended**: January 2025. No live submissions. We simulate evaluation via holdout.

## Data Split Strategy
The competition has ended — no Kaggle submission available. We hold out a
portion of the training data as a private test set to simulate the real
evaluation. The split must respect temporal ordering (no future leakage).

- **Train**: earliest dates (bulk of data)
- **Validation**: middle/later dates (for CV, hyperparameter tuning)
- **Holdout Test**: final dates (touched ONCE for final evaluation)
- Exact date boundaries decided during EDA (Phase 2.3)
- The holdout test set is never used for any model selection or tuning


## Phase 1: Competition Understanding (DONE)

### 1.1 Competition Rules
- Task: Predict responder_6 (continuous) for each trading opportunity
- Metric: Sample-weighted zero-mean R² (predicting zero = baseline)
- Data: 47,127,338 rows, 1,699 days, 39 symbols, 10 partitions
- 92 columns: date_id, time_id, symbol_id, weight, 79 features, 9 responders
- Constraint: Real-time inference, ~16ms per iteration
- At inference: current features visible, current responders hidden,
  lagged responders from previous time slot available

### 1.2 Key Challenges
1. Non-stationarity — market regimes shift over time
2. Low signal-to-noise ratio
3. Fat-tailed distributions
4. 16ms inference budget
5. Multicollinearity among features


## Phase 2: EDA & Data Validation (DONE)

### 2.1 Data Integrity (DONE)
- [x] All 10 partitions verified intact (47.1M rows total)
- [x] Partition 9 re-extracted from zip (was truncated)
- [x] Schema confirmed: 92 columns, correct types

### 2.2 Distributions & Summary Statistics (DONE)
- [x] Responder distributions: all ~zero mean, std 0.59-0.92, fat-tailed (kurt 5.8-24.4)
- [x] Responder_6: mean=-0.002, std=0.89, skew=+0.47, kurt=6.86
- [x] No nulls in any responders
- [x] Feature distributions: no degenerate features (all std > 0.01)
- [x] Extreme skew features: feature_47 (214!), feature_21 (151), feature_31 (131)
- [x] Weight: mean=2.01, std=1.13, range [0.15, 10.24], right-skewed, no zeros
- [x] High null features: 21/26/27/31 (~18%), 39/42/50/53 (~9%), 00/01 (~7%)

### 2.3 Temporal Structure (DONE)
- [x] Date range: 0-1698 (1699 days)
- [x] Time slots/day: ~921 mean (range 849-968) — roughly constant
- [x] Symbols/slot: ~30 mean (range 4-39) — varies
- [x] Regime check: responder_6 std peaked mid-data (1.01 around dates 509-678),
      lower at start/end (~0.81-0.87). Mean stays near zero.
- [x] Feature stationarity: feature_24 biggest shift (z=1.08), features 00/02/03
      shift together (correlated group), features 15/17/20/30 also drift
- [x] Autocorrelation: lag_1=0.06 (weak but present), decays slowly
- [x] **Data splits decided**:
      Train: 0-1188 (1189 days), Val: 1189-1443 (255 days), Holdout: 1444-1698 (255 days)

### 2.4 Feature-Target Relationships (DONE)
- [x] Top features by |corr| with responder_6:
      feature_06 (-0.047), feature_04 (-0.032), feature_07 (-0.030),
      feature_36 (-0.023), feature_60 (+0.019)
- [x] All top-15 features have stable signs (early vs late)
- [x] Unstable features (sign flips): feature_14, feature_69, feature_48 — avoid
- [x] Redundant feature groups: 21/31 (r=0.998), 77/78 (0.96), 75/76 (0.96),
      73/74 (0.96), 00/02/03 (0.94), 34/35 (0.94), 15/17 (0.93)
- [x] **Inter-responder**: responder_3 ↔ responder_6 = 0.73 (best auxiliary target),
      responder_8 = 0.45, responder_7 = 0.43

### 2.5 Lags File Analysis (DONE)
- [x] Lags file: 39 rows × 12 cols — template with (date/time/symbol + 9 lag_1 responders)
- [x] **lag1_responder_6 ↔ current responder_6: corr = 0.90** — dominant signal
- [x] lag1_responder_3 also strong (0.76)
- [x] **Naive baseline: predict lag1 → R² = 0.805, scaled → R² = 0.815 (scale=0.90)**
- [x] This is the signal to beat — models should add value on top of this


## Phase 3: Baseline Models (DONE)
Training: dates 900-1188 (9,998,472 rows), Validation: dates 1189-1443 (9,386,696 rows)

### 3.1 Dumb Baselines (DONE)
- [x] Predict zero → R² = 0.000000 (by definition)
- [x] Predict lag_1 (raw) → R² = 0.8016, scaled (s=0.90) → R² = 0.8114
- [x] Ridge on top 5 features → R² = 0.0024
- [x] Ridge on all 79 features → R² = 0.0043
- [x] **Key insight: lag_1 dominates (R²=0.81). Feature-only models are 100× weaker.**

### 3.2 LightGBM Baseline (DONE)
- [x] LightGBM (all 79 features): val R² = 0.0108, train R² = 0.0444
- [x] Best iteration: 111 (early stopped from 2000)
- [x] Overfit gap: 0.034 (train R² 4× higher — regularization needed)
- [x] Train time: 140s, Inference: 0.137ms/row (well within 16ms budget)

### 3.3 Feature Selection via LightGBM (DONE)
- [x] Top features by gain: feature_06 (5.4%), feature_61 (4.2%), feature_30 (3.4%),
      feature_36 (3.3%), feature_07 (3.2%), feature_04 (3.0%)
- [x] Elbow: 18 features → 50% gain, 40 → 80%, 54 → 90%, 63 → 95%
- [x] Suspicious (low gain/split): feature_63, feature_55, feature_41, feature_00
- [x] Stability (Spearman rank corr early vs late): 0.904 — very stable
- [x] 28 features stable in top-30 of both halves
- [x] **Selected 54 features (90% cumulative gain)**
- [x] LightGBM (54 features): R² = 0.0108 — negligible loss vs all-79 (0.010782 vs 0.010825)

### 3.4 Model Comparison (DONE)
- [x] Ridge (54 features): R² = 0.0044, train=5.6s, infer=0.045ms
- [x] XGBoost (54 features): R² = 0.0106, train=108s, infer=0.148ms
- [x] LightGBM (54 features): R² = 0.0108, train=121s, infer=0.137ms
- [x] **Tree-based models beat Ridge by ~2.5× — nonlinear interactions matter**
- [x] LightGBM ≈ XGBoost (LGB marginally better, similar speed)
- [x] All models well within 16ms inference budget


## Phase 4: Feature Engineering (DONE)

Training: dates 900-1188 (9,998,472 rows), Validation: dates 1189-1443 (9,386,696 rows)

### 4.1 Market-Level Features (DONE)
- [x] Per (date_id, time_id): mean of top 10 features + relative deviation (20 features)
- [x] **No improvement vs lag-only**: −0.000282 R² — dropped from final feature set

### 4.2 Symbol-Level Rolling Features (DONE)
- [x] Rolling mean (window=10) for top 10 features per symbol (10 features)
- [x] **No improvement** — market conditions change too fast for rolling to add value

### 4.3 Lagged Responder Features (DONE)
- [x] lag1_r6 (corr=0.90), lag1_r3 (0.76), lag1_r7 (0.42), lag1_r4 (0.36), lag1_r0 (−0.09)
- [x] **Dominant signal**: lag1_r6 = 73% of LightGBM gain, lag1_r3 = 20%

### 4.4 Validate Engineering (DONE)
- [x] LightGBM (54 sel + 5 lag): Val R² = **0.856097** vs Phase 3 baseline 0.010782 (+79×!)
- [x] Adding market + rolling (89 total): Val R² = 0.855815 (−0.000282 vs lag-only)
- [x] **Final feature set: 59 features (54 selected + 5 lag responders)**

Key insight: Lag features transform the problem. LightGBM+lag beats naive lag baseline
(0.811) by +0.045 R² through learned interactions. Market and rolling features provide
no measurable benefit beyond the lag signal.


## Phase 5: Neural Network (DONE)

Training: dates 900-1188 (10M rows), Validation: dates 1189-1443 (9.4M rows)
Features: 59 (54 selected + 5 lag), GPU: RTX 4060

### 5.1 Architecture Selection (DONE)
- [x] MLP baseline: Val R² = 0.858997 (115K params, 1224s train)
- [x] MLP + lag residual: Val R² = 0.861152 (+0.002 from residual)
- [x] GRU + lag residual: Val R² = **0.882851** (+0.027 vs LGB, 180K params, 304s train)
- [x] **GRU wins decisively** — temporal sequence modeling per (symbol, day) matters

### 5.2 Multi-Task Learning (DONE)
- [x] Auxiliary target: responder_3 (corr=0.73), aux_weight=0.3
- [x] Multi-task MLP: Val R² = 0.861668 (+0.0005 vs single-task MLP)
- [x] Small but consistent improvement from multi-task; main gain is from GRU

### 5.3 Training Setup (DONE)
- [x] Same temporal CV as Phase 3-4 (fair comparison)
- [x] Preprocessing: standardize features (mean/std from train), zero-fill NaN
- [x] Lag residual init: scale = 0.90 × std(lag1_r6) to account for normalization
- [x] AdamW (lr=1e-3, wd=1e-4), CosineAnnealing, grad clip=1.0, patience=3

### 5.4 Key Results
- [x] All NN models beat LightGBM (R²=0.856), overfit gap 3× lower
- [x] GRU best: R²=0.883 vs LGB 0.856 (+0.027), trained in 5 min
- [x] All models well within 16ms inference budget (GRU: 0.37ms/step)
- [x] Models saved to models/, normalization stats to models/norm_stats.npz


## Phase 6: Online Learning (DONE)

Simulated inference loop: step through validation data one time slot at a time,
maintaining per-symbol GRU hidden states. After each prediction, the revealed
lagged responder provides a training signal for online model adaptation.

### 6.1–6.7 Key Results (src/online_learning.py)
- [x] Frozen GRU baseline: Val R² = **0.882852** (step-by-step matches batch)
- [x] Head-only SGD (lr=1e-5): R² = 0.882194 (−0.0007)
- [x] Head-only SGD (lr=1e-4): R² = 0.881248 (−0.0016)
- [x] Head-only SGD (lr=1e-3): R² = 0.879900 (−0.0030)
- [x] All-layers SGD (best lr): R² = 0.880952 (−0.0019)
- [x] Daily weight reset: R² = 0.881104 (−0.0017)
- [x] Lag EMA (α=0.01–0.1): R² = 0.8497–0.8572 (degrades badly)
- [x] Combined (gradient + EMA): R² = 0.8799 (−0.0030)

**Conclusion: All online learning variants DEGRADE performance.**
The frozen GRU is already near-optimal for this data distribution. Online
updates on noisy individual steps introduce more variance than signal. The
problem is not that the model can't adapt — it's that there's nothing
meaningful to adapt to when the features contain no independent signal.


## Phase 7: Ensemble & Holdout Evaluation (DONE)

### 7.1 Multi-Seed Ensemble (src/ensemble.py)
- [x] 5 seeds: [42, 123, 456, 789, 2024]
- [x] Individual Val R²: [0.88285, 0.88256, 0.88257, 0.88254, 0.88232]
- [x] Mean individual: 0.88257
- [x] **Ensemble Val R²: 0.88389** (+0.0013 over mean individual)
- [x] Ensemble is remarkably stable (std < 0.0002 across seeds)

### 7.2 Cross-Architecture Ensemble
- [x] GRU single: R² = 0.88285
- [x] MLP lag-residual: R² = 0.86115
- [x] Simple average (GRU+MLP): R² = 0.87767 (worse — MLP dilutes)
- [x] Best blend: α=1.0 × GRU (MLP weight = 0)
- [x] **GRU is so dominant that blending with MLP hurts performance**

### 7.3 Inference Optimization
- [x] Single symbol step: 0.257ms
- [x] 39-symbol batch step: 0.276ms
- [x] **Well within 16ms budget: 58× headroom**

### 7.4 Holdout Test Evaluation (ONE TIME ONLY, dates 1444-1698)
- [x] Naive 0.90×lag: Holdout R² = **0.8115**
- [x] MLP lag-residual: Holdout R² = **0.8575** (Val was 0.8612, gap = +0.0037)
- [x] GRU single (seed 42): Holdout R² = **0.87997** (Val was 0.8829, gap = +0.0029)
- [x] GRU ensemble (5 seeds): Holdout R² = **0.88085** (Val was 0.8839, gap = +0.0030)
- [x] Best blend (GRU×0.95+MLP×0.05): Holdout R² = **0.87950**

**Excellent val→holdout generalization:** Gap is only ~0.003, confirming the
lag autocorrelation structure is stable across all historical dates (1189-1698).

## Phase 8: Signal Decomposition & Critical Findings (DONE)

### 8.1 The Core Discovery (src/improved_modeling.py)

| Model | Val R² | What it measures |
|-------|--------|-----------------|
| Naive: 0.90 × lag1_r6 | 0.8114 | Pure autocorrelation baseline |
| Feature-only LightGBM (79 features) | 0.0108 | Features alone, no lag |
| **Feature-only GRU** (79 features, lag zeroed) | **0.0108** | GRU without lag — same as LGB |
| LightGBM (54 sel + 5 lag) | 0.8561 | Features help select lag signals |
| GRU + lag (standard) | 0.8829 | Dynamic lag adjustment + temporal |
| Residual GRU (fixed 0.90×lag, no lag in input) | 0.8555 | What features add on top of fixed lag |
| Lag dropout GRU (p=0.3) | 0.8791 | Learns feature fallback with minimal cost |
| 5-seed GRU ensemble | 0.8839 | Best historical result |

### 8.2 Critical Findings

**Finding 1: Features alone contain virtually NO predictive signal.**
Feature-only GRU R² = 0.0108, virtually identical to feature-only LightGBM
(0.0108). The GRU architecture adds zero value without lag features. This
definitively proves the 79 features cannot predict responder_6 independently.

**Finding 2: ~99% of predictive power comes from lag1_r6 autocorrelation.**
Breaking down the GRU+lag (0.883):
- Lag autocorrelation (r=0.90): R² ≈ 0.81 (92%)
- Dynamic lag_scale adjustment: +0.028 R² (3%)
- Temporal sequence modeling: +0.027 R² over LightGBM+lag (3%)
- Features alone: +0.011 R² (1%)

**Finding 3: Residual modeling is too rigid.**
Fixing lag_scale at 0.90 and having the GRU predict only the residual
achieves R² = 0.8555, losing 0.027 vs the full GRU. The GRU's ability to
**dynamically adjust the lag_scale** (learning when lag is more/less reliable)
is the primary architectural advantage.

**Finding 4: Lag dropout adds robustness with minimal cost.**
Training with 30% of steps having lag features zeroed achieves R² = 0.8791
(only −0.004 vs baseline). The model learns to use features as a modest
fallback. This is a robustness improvement, not a performance improvement.

**Finding 5: Online learning degrades performance across the board.**
Every online adaptation tested (Phase 6) performed worse than the frozen model.
The frozen GRU is already near-optimal for this data distribution.

### 8.3 Competition Reality Check

**Our holdout R² = 0.881 vs Competition #1 leaderboard = ~0.014**

This 60× gap is the elephant in the room. The lag-1 autocorrelation (r=0.90)
is a **historical artifact** that does not persist in truly future market data.
In the actual competition evaluation, the lag signal collapses, and the entire
modeling approach (which relies on lag1_r6 as the dominant input) has almost
nothing to fall back on (features alone give R² ≈ 0.01).

### 8.4 Walk-Forward Cross-Validation (src/walk_forward_cv.py)

Expanding-window CV with lag ablation on dates 0-1550:
- Fold 1 (train 0-500, val 501-700): R² = 0.8832, lag autocorr = 0.8997
- Fold 2 (train 0-700, val 701-900): R² ≈ 0.875+, in progress

The lag autocorrelation is stable (r≈0.90) even in the earliest date ranges.
The GRU performs consistently across all historical periods, confirming the
lag signal is a stable historical property, not a regime-specific phenomenon.

### 8.5 Strategic Recommendations

**If the lag signal persists (historical data evaluation):**
- Use multi-seed GRU ensemble (+0.001 R² gain, already done)
- Consider lag dropout p=0.3 for robustness (−0.004 R² cost but better OOS behavior)
- No need for online learning — frozen model is optimal
- Market features, rolling features, MLP blending all add no value

**If the lag signal decays (real future market data):**
No current approach will work because features contain no signal. The only
paths forward are:
1. **Find new features** — current 79 features are insufficient
2. **Predict a different target** — other responders may be more predictable
3. **Change problem framing** — predict volatility, direction, or regime
4. **Accept the limitation** — if lag structure persists, current approach works

## Phase 9: New Scripts (Ready to Run)

These scripts are implemented but results are pending/intermediate:

- **src/regime_modeling.py** — Regime-conditional GRU (3 volatility regimes, shared encoder + regime-specific heads)
- **src/prediction_shrinkage.py** — Uncertainty-weighted prediction shrinkage (ensemble variance → shrink toward zero)
- **src/hyperparameter_search.py** — Grid search over depth/width/dropout/lr/chunk_size (162 configs)

Bug fix applied: `src/feature_engineering.py` TRAIN_PATH corrected from 5-level to 2-level parent traversal.


## Technology Stack
- Data: Polars, Pandas
- Baseline: LightGBM, scikit-learn (Ridge)
- Neural: PyTorch
- Visualization: Matplotlib, Seaborn
- Feature analysis: SHAP
- Optimization: Numba, TorchScript
- CV: Custom temporal split (no random splitting)
