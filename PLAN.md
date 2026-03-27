# Jane Street 2024 Kaggle Competition Plan (v2)

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


## Phase 5: Neural Network

### 5.1 Architecture Selection
- [ ] Start with simple MLP baseline on selected features
- [ ] Then GRU — treat one day as a sequence
- [ ] Compare MLP vs GRU on same CV — does sequence modeling help?
- [ ] Determine number of layers, hidden units via CV (don't pre-assume)

### 5.2 Multi-Task Learning
- [ ] Use EDA Phase 2.4 to decide which responders to use as auxiliary targets
- [ ] Compare single-task (responder_6 only) vs multi-task → measure CV delta
- [ ] Tune auxiliary loss weights via CV

### 5.3 Training Setup
- [ ] Decide training date range from EDA (stationarity analysis)
- [ ] Preprocessing: standardize features, zero-fill NaN
- [ ] Same temporal CV as Phase 3 for fair comparison
- [ ] Experiment log: track every run (hyperparams, CV score, notes)

### 5.4 Alternative Architectures (if time permits)
- [ ] Smaller/larger GRU variants
- [ ] MLP with residual connections
- [ ] Compare all architectures on same CV


## Phase 6: Online Learning

### 6.1 Baseline Online Learning
- [ ] Implement basic version: update on responder_6_lag_1 only,
      all layers, fixed lr, SGD
- [ ] Measure CV delta vs frozen model

### 6.2 Target Selection
- [ ] Test update on responder_6 only vs multiple responders vs all 9
- [ ] If multi-task: tune relative weights of each responder's loss
- [ ] Use Phase 2.4 correlations to guide which responders to try first

### 6.3 Layer Selection
- [ ] Test: update all layers vs last layer only vs last N layers
- [ ] Measure CV and stability for each

### 6.4 Learning Rate & Optimizer
- [ ] Sweep lr (log scale, range TBD)
- [ ] Compare SGD vs Adam for online updates
- [ ] Test fixed lr vs decaying lr within a day

### 6.5 Feature-Based Adaptation (non-gradient)
- [ ] Add running statistics of lagged responders as input features
      (EMA of recent responder values, rolling std)
- [ ] Test: gradient-only vs feature-only vs both combined
- [ ] Update feature normalization statistics online
      (running mean/std to handle feature distribution shift)

### 6.6 Day Boundary Handling
- [ ] Test: keep adapted weights vs reset each morning
- [ ] Use EDA (Phase 2.3) to inform — do regimes persist across days?
- [ ] Test: reset GRU hidden state at day boundary vs carry forward

### 6.7 Validation & Stability
- [ ] Simulate full inference loop on validation days
- [ ] Monitor: does model destabilize after 500+ updates?
- [ ] Verify total inference time (update + predict) < 16ms
- [ ] Test on multiple validation periods to confirm robustness


## Phase 7: Ensemble & Final Evaluation

### 7.1 Multi-Seed Ensemble
- [ ] Train N seeds per best architecture
- [ ] Simple average of predictions
- [ ] Measure ensemble CV vs best single model

### 7.2 Cross-Architecture Ensemble (if multiple models are competitive)
- [ ] Blend different model types
- [ ] Simple average or learned weights on validation set

### 7.3 Inference Optimization
- [ ] Profile end-to-end inference time per row
- [ ] If > 16ms: TorchScript, reduce model size, Numba for preprocessing

### 7.4 Holdout Test Evaluation
- [ ] Run final model on holdout test set (ONE time only)
- [ ] Report weighted R² on holdout
- [ ] Compare holdout R² vs validation R² — if large gap, investigate
- [ ] Document final results and lessons learned


## Phase 8: Iteration
- [ ] Analyze holdout results — where does the model fail?
- [ ] Revisit feature engineering, training ranges, architectures
- [ ] Re-evaluate on validation set (not holdout) for further tuning


## Decision Points (decided by data, not pre-assumed)
- Train/validation/holdout date boundaries → Phase 2.3
- Which features to keep → Phase 3.3
- Which responders are useful as auxiliary targets → Phase 2.4
- What training date range to use → Phase 2.3
- GRU vs MLP vs other → Phase 5.1
- Online learning rate → Phase 6.4
- Online learning targets → Phase 6.2
- Online learning layers → Phase 6.3
- Gradient vs feature-based adaptation → Phase 6.5
- Day boundary reset policy → Phase 6.6
- Ensemble composition → Phase 7


## Technology Stack
- Data: Polars, Pandas
- Baseline: LightGBM, scikit-learn (Ridge)
- Neural: PyTorch
- Visualization: Matplotlib, Seaborn
- Feature analysis: SHAP
- Optimization: Numba, TorchScript
- CV: Custom temporal split (no random splitting)
