# Project Redesign — 2026-04-17

**Supersedes** `RedesignPlan_04-13.md`. That plan correctly identified the
central issue (Val R² ≈ 0.88 vs. Kaggle LB ≈ 0.014 = 60× gap likely driven by
CV leakage + single-feature dependence) but responded with an academic
enforcement apparatus that does not fit a competition workflow. This revision
keeps the 04-13 plan's correct instincts (walk-forward CV, non-lag R² as a
first-class metric, FOLDS_A/FOLDS_B role separation, inner-val early stopping,
fold-local selection) and fixes the following:

- **Public Kaggle LB is the primary ground truth**, not a local holdout
  slice. The previous plan never mentioned the LB. It is the only
  out-of-sample signal that is genuinely unseen; everything on our
  workstation can be peeked at by construction.
- **Scoring-formula verification is Day 1, highest-EV task.** A zero-mean
  vs. sample-mean R² mismatch, or a weight-aggregation bug, can produce a
  10× score gap by itself and invalidates every number below until fixed.
- **lag1_r6 mechanism investigation precedes modeling.** The 99%-power
  finding may be a pipeline off-by-one (fixable in minutes) rather than a
  generalization failure. The 04-13 plan assumed the latter.
- **Sealed storage theater removed.** On a single-user workstation with no
  root, `chmod 700` and phase tokens are advisory, not enforcing. Honor
  system is declared explicitly so guarantees match infrastructure.
- **`NLF_A_ref` replaced with `NLF_best3`** — the max non-lag R² across
  {ridge, LightGBM, small MLP} on FOLDS_A. One-model reference biases the
  floor by architecture family; best-of-three is defensible regardless of
  which family wins.
- **`σ_min` derived from FOLDS_A R² variance**, frozen in `splits.py`
  before FOLDS_B is read. Predeclared `0.02` was inherited from the
  protocol the redesign declared untrustworthy.
- **Ship-with-writeup replaces exit-on-fail.** Competitions have
  deadlines; submitting nothing guarantees last place. The discipline
  "don't retune after seeing holdout" is preserved; the consequence
  "submit nothing" is not.
- **Cross-fold aggregation permitted under fit-A-apply-B rule.** Blanket
  ban forces fold 1 (train 0–500) to select features from 500 days of
  data; that materially hurts early-fold quality with no leakage
  protection win, because the real leakage channel is *applying* an
  aggregated set back onto the folds that fed it.
- **Phase 0 compressed to ~3 days** (from 2 weeks). The elaborate
  provenance chain, append-only ledger, and fixture test suite are
  replaced with a 50-line leakage checker covering the three real
  failure classes (normalize-on-val, lag-off-by-one, time_id/date_id
  swap). The heavy version is valuable in a team setting; for a single
  researcher it is a distraction.
- **Microstructure / cross-sectional investigation elevated to Phase 1.**
  This is where non-lag signal actually lives in HFT-style data and was
  absent from the 04-13 plan.
- **Submission budget is the real enforcement mechanism.** Predeclare
  ~10 submissions across 10 weeks; each one is a one-shot read of a
  genuinely unseen distribution. CV↔LB tracking is the honest validation.

---

## Guiding Principles

1. **LB is truth.** CV tracks LB, not the reverse. If they diverge by
   more than `2 · σ_min`, investigate CV before trusting more CV numbers.
2. **Cheapest diagnostics first.** Scoring formula, lag1 mechanism, one
   LB submission of the current model — all before any infrastructure.
3. **Submission budget is the real gate.** Kaggle allows ~5/day; we
   allow ourselves ~10 over the cycle. Each read is one-shot.
4. **Ship always.** A failed holdout gate produces a writeup and a
   submission, not a no-ship. The submission with the failure noted is
   always strictly better than nothing.
5. **Protocol discipline in service of science, not of protocol.** If a
   rule is costing more than it saves, revise the rule.
6. **Leakage-clean by construction.** Train-only fits; val/holdout get
   frozen parameters. Enforced by a small audit script.
7. **Honor system where mechanisms cannot enforce.** We label every
   guarantee with the mechanism that backs it. "Sealed" is not a word
   this plan uses for anything on our workstation.

---

## Phase 0 — Minimal Infrastructure (3 days, blocking modeling)

### 0.1 Scoring formula validation (Day 1, highest EV)

- Pull Kaggle's metric code from the competition's evaluation page
  (`https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview/evaluation`).
- Implement `weighted_r2(y, yhat, w)` in `src/metrics.py`.
- **Unit tests**: construct 3–5 synthetic cases (including edge cases:
  all-zero predictions, perfect predictions, negative R², zero-weight
  rows) and verify our implementation matches Kaggle's byte-for-byte.
- If the old 0.88 was computed under a different formula than Kaggle's,
  flag it in the report. This alone may explain much of the 60× gap.

### 0.2 Splits and folds (Day 2)

- `src/splits.py`, single source of truth:
  ```python
  TRAIN_RANGE   = (0, 1188)
  VAL_RANGE     = (1189, 1443)
  HOLDOUT_RANGE = (1444, 1571)    # local one-shot dev holdout
  RESERVE_RANGE = (1572, 1698)    # untouched this cycle; honor system

  FOLDS_A = [   # tuning + all pre-holdout decisions
      (train=(0, 500),   val=(501, 700)),
      (train=(0, 700),   val=(701, 900)),
      (train=(0, 900),   val=(901, 1100)),
      (train=(0, 1100),  val=(1101, 1300)),
  ]
  FOLDS_B = [   # outer eval, one-shot at Phase 6.1
      (train=(0, 1300),  val=(1301, 1443)),
  ]

  def inner_split(train_lo, train_hi, frac=0.10):
      """Tail ~10% of train is inner-val for early stopping."""
      cut = train_hi - int(frac * (train_hi - train_lo))
      return (train_lo, cut), (cut + 1, train_hi)

  # σ_min is BLANK at project start; populated by freeze_sigma_min.py
  # after Phase 2 baselines complete, BEFORE any FOLDS_B read.
  SIGMA_MIN = None
  ```
- `src/freeze_sigma_min.py`: reads `reports/baselines/*.json`,
  computes `std(FOLDS_A mean R²)` across baseline architectures,
  writes the number back into `splits.py`, commits it. Runs exactly
  once at the end of Phase 2. The audit script (§0.4) refuses to
  authorize Phase 6 until `SIGMA_MIN` is populated and committed.

### 0.3 Metrics, data loader, and provenance-lite (Day 2)

- `src/metrics.py`: `weighted_r2`, `cv_score(fold_results, lam=0.5)`,
  `non_lag_r2(model, X, y, w, lag_cols, mode={"zero","noise"})`.
- `src/data.py` (rewrite): every `load(...)` returns `(df, manifest)`
  where manifest records `{date_range, n_rows, n_cols, sha256}`. No
  hard-coded date cutoffs. Every report saves the manifests it used.
- `load_holdout()` and `load_reserve()` **exist but raise** by default.
  They only return data when `JS_PHASE_UNLOCK=HOLDOUT_PHASE6` (or
  `=RESERVE`) is set in the environment. This is the honor-system
  mechanism — it doesn't prevent reads; it makes accidental reads
  loud. Combined with the submission-budget tracker, that is
  sufficient enforcement for a single-researcher project.

### 0.4 Minimal leakage audit (Day 3)

- `src/audit_leakage.py`, ~50 lines, three concrete checks:
  1. **Normalize-on-val**: any transform with `.fit()` state must have
     been fit with `date_id < val_start` in its provenance manifest.
  2. **Lag off-by-one**: at `(date_id=d, time_id=t)`, `lag1_r6` equals
     the previous slot's `responder_6` for the same symbol. Spot-check
     on first/last time_id of every val `date_id` (boundary slots
     where look-ahead bugs hide) + random 5%.
  3. **time_id vs date_id swap**: a fixture where two rows at same
     `date_id` but different `time_id` must not produce identical
     regime/rolling features.
- Exits non-zero on violation. Runs in CI on every pipeline module.
- Not in scope for the slim version: append-only FOLDS_B ledger,
  provenance-chain walker, cross-pool artifact tracking, deterministic
  leakage fixture suite. If a team scales up, re-introduce these.
- The **FOLDS_B read discipline** is enforced by a single line in
  `data.py`: `load_folds_b_val()` raises unless
  `JS_PHASE_UNLOCK=FOLDS_B_PHASE6`. Grep the repo for that constant
  at PR time; that's the audit.

### 0.5 Submission tracker (Day 3)

- `reports/lb_tracker.md`: append-only table of `{date, submission_id,
  CV_score, LB_score, notes}`. Updated by hand after each submission.
- Predeclared budget: **10 submissions total** across the 10-week
  cycle. Reasoning: Kaggle's public LB is itself noisy and leaks
  information about the test distribution; we want to use it as a
  sparse ground-truth signal, not a continuous tuning loop.
- CV↔LB correlation computed every 3 submissions. If `|CV − LB| >
  2 · σ_min` at any submission, **pause modeling** and investigate
  CV before spending more of the budget.

### Phase 0 Gate

- Scoring formula matches Kaggle's on all unit tests.
- Audit script exits clean against `data.py`.
- CV folds produce identical indices across reruns.
- `splits.py` committed; `SIGMA_MIN = None` is explicit.
- `lb_tracker.md` initialized.

---

## Phase 1 — Diagnostics Before Modeling (1 week)

**Premise:** Before rebuilding, use cheap diagnostics to decide what
actually needs rebuilding. The 04-13 plan jumped from "old numbers
suspicious" to "rebuild everything"; a top QR spends a week finding
out *which parts* of the old work were wrong and *why*.

### 1.1 LB submission of the current best model (Day 1–2, 1 submission)

- Take the ensemble model that produced the old 0.881 on val. Submit
  it to Kaggle, unmodified, as the competition's scoring-notebook
  format requires.
- Record the LB score.
- **Interpretation rubric:**
  - LB ≈ 0.88: the old CV was valid; the redesign's premise is wrong;
    stop and ship. (Lowest probability.)
  - LB ≈ 0.01–0.05: gap confirmed; redesign proceeds.
  - LB is negative or 0: something is badly broken beyond CV
    leakage (scoring format mismatch, input column order, etc.);
    debug the submission pipeline before any modeling.
- This submission is ~1 hour of work and removes weeks of
  speculation. Budget cost: 1/10.

### 1.2 lag1_r6 mechanism investigation (Day 2–3)

- Open `lags.parquet` provided by the competition. Trace exactly how
  `lag1_r6` is constructed in our pipeline.
- Key questions:
  - Is `lag1_r6` at `(date_id=d, time_id=t)` the responder from
    `(date_id=d-1, time_id=t)` for the same symbol? Or the previous
    slot? The competition's convention matters.
  - Does the 0.90 autocorrelation hold on dates 0–899 (which the
    04-13 EDA skipped) and on the private test rows visible through
    `lags.parquet`?
  - Does the autocorrelation persist after winsorizing `responder_6`
    at ±6σ? If it disappears, the signal is tail-driven.
- **Two distinct hypotheses we're separating:**
  1. **Pipeline bug**: our `lag1_r6` column is not actually last
     slot's responder due to a shift error; correlation is spurious.
     Fix = one-line change, ship.
  2. **Genuine autocorrelation**: lag1_r6 really is the prior
     responder and really has 0.90 correlation on training data,
     but Kaggle's live test data has that autocorrelation permuted
     away (likely: test data is shuffled at date level).
     Fix = accept the gap as a structural feature of the evaluation
     and focus non-lag work.

### 1.3 Full-range EDA (Day 3–5)

Rebuild the 04-13 Phase 1 EDA, **but only on TRAIN + VAL (0–1443)**.
HOLDOUT (1444–1571) stays locked (honor system, §0.3) until Phase 6.

- Responder_6: mean, std, skew, kurtosis per 100-day block on 0–1443.
- Weight distribution and null rates per block.
- Autocorrelation of responder_6 at lag 1/5/20, separately on 0–899
  and 900–1443, to test whether the old 900-cutoff decision had any
  basis in distributional drift.
- Per-symbol activity matrix: do symbols enter/leave? (Relevant for
  ensemble/regime design.)
- Save each analysis with a `LoadManifest` (§0.3).

### 1.4 Microstructure and cross-sectional investigation (Day 5–7, NEW)

This was absent from 04-13 and is where genuine non-lag signal lives
in HFT-style data.

- **Intraday structure**: bin `time_id` into quintiles; measure mean
  |responder_6| per bin. Is there a time-of-day effect?
- **Cross-sectional rank**: at each `(date_id, time_id)`, rank the
  39 symbols by responder_6. Compute rank autocorrelation slot-to-slot.
  If symbols that did well last slot continue to do well this slot,
  that's cross-sectional momentum — a candidate feature family.
- **Market-mean coupling**: at each `(date_id, time_id)`, compute
  mean responder across symbols. Does individual responder_6 beta to
  this mean? That's a market-wide factor; residualizing may expose
  idiosyncratic signal.
- **Weight column interpretation**: correlate `weight` with
  `|responder_6|`, `std(responder_6 over recent window)`, and
  feature columns. What is `weight` actually proxying? This changes
  how we should train.

### Phase 1 Gate

- LB baseline score recorded in `lb_tracker.md`.
- lag1_r6 mechanism documented: pipeline-correct or pipeline-bug.
- At least one cross-sectional hypothesis identified for Phase 3.
- No model training has happened yet.

---

## Phase 2 — Honest Baselines (1 week)

Goal: under the new protocol, produce reference numbers that all later
work must beat. Populate `SIGMA_MIN` at the end of this phase.

### 2.1 Dumb baselines (walk-forward on FOLDS_A)

- Predict 0 → R² = 0 by construction.
- Predict `k · lag1_r6` with `k` fit per-fold on that fold's train
  only. Report mean and std across folds.

### 2.2 Linear baselines

- Ridge on 79 features + lag responders.
- Ridge on `responder_6 − k̂ · lag1_r6` (residual). **This is the
  single most informative baseline**: if ridge-on-residual R² is
  approximately 0, features add nothing beyond lag1, and the ceiling
  on non-lag modeling is near zero regardless of architecture.

### 2.3 Tree baselines

- LightGBM and XGBoost on full feature set + lags. Same
  hyperparameters across folds (so cross-fold variance measures
  data variance, not tuning variance).
- Report per-fold R² and feature importances. Importances are
  **descriptive only** — they may inform hypotheses but do not
  drive CV feature sets (§2.5).

### 2.4 Non-lag reference (`NLF_best3`)

- Train three feature-only models (lag columns zero-ablated from
  input): ridge, LightGBM, small MLP.
- Same FOLDS_A, same tuning budget per model.
- `NLF_best3 = max over (ridge, lgbm, mlp) of (mean − 0.5·std on FOLDS_A)`.
- Record `std(NLF_best3_across_folds)` = ε_nl.
- **Acceptance threshold for later architectures** = `NLF_best3 + ε_nl`
  on FOLDS_A non-lag R². This is the non-lag floor used by §4.4, §6.3,
  and every "Definition of Ship" reference to non-lag R².
- If `NLF_best3 ≤ 0.01` on FOLDS_A, non-lag signal is effectively
  absent and the cycle's realistic ceiling is a lag-dominated model
  with structural honesty rather than a genuine feature-learning win.
  Document this finding; proceed with Phase 3 but with reduced ambition.

### 2.5 Fold-local feature selection, corrected rule

The 04-13 blanket ban on cross-fold aggregation was over-corrected.
The real leakage rule is:

- **Cross-fold aggregation is allowed** as a feature-selection input
  if and only if the aggregated set is applied **only to folds or
  data that did not contribute to the aggregation**.
- In practice: a "top-K by mean gain across FOLDS_A folds 1–4" set
  may be used for (a) the FOLDS_B evaluation run, and (b) the final
  holdout retrain on 0–1443. It may **not** be used to re-score
  FOLDS_A folds 1–4.
- For FOLDS_A scoring itself, each fold picks its own features from
  its own training data only — as 04-13 required.
- This preserves the disjoint-pool discipline while allowing pooled
  selection for outer eval. It also matches how feature selection
  actually works in production quant pipelines.

### 2.6 Freeze SIGMA_MIN

- Run `freeze_sigma_min.py`. It computes
  `σ_min = max(std(FOLDS_A R²) across baselines, 0.005)` —
  the max guards against a pathologically stable early phase
  setting σ too tight.
- Writes back into `splits.py`; commits. `SIGMA_MIN` is now frozen
  for the rest of the cycle.

### Phase 2 Gate

- `NLF_best3` and `ε_nl` recorded.
- `SIGMA_MIN` frozen in `splits.py`, committed.
- Best tree baseline's FOLDS_A score beats `k · lag1_r6` by at
  least `0.5 · σ_min`, or Kill Criterion K1 triggers.

---

## Phase 3 — Feature Engineering (2 weeks)

Goal: find non-lag signal. If `NLF_best3 ≤ 0.01` from Phase 2, this
phase's realistic ceiling is constrained; document target before starting.

### 3.1 Lag family refinements (Week 1, Day 1–2)

- Multi-horizon lag smoothing: mean of `lag1_r6` over last `k ∈ {3, 5, 10}`
  intraday slots. Tests whether smoothing stabilizes the lag signal.
- Multi-lag: `lag1_r0…r8` (beyond just `r6`). Do other responders
  help predict `r6`?

### 3.2 Cross-sectional features (Week 1, Day 3–5, NEW priority)

Directly follows from Phase 1.4 investigation.

- At each `(date_id, time_id)`: cross-sectional rank of features among
  the 39 symbols.
- Market-mean residuals: `feature_i − mean(feature_i across symbols at
  same (date_id, time_id))`.
- Cross-sectional momentum: own-symbol responder rank at previous slot.
- **All aggregates shifted by ≥ 1 time_id within day** to prevent
  same-slot leakage. Verified by audit script.

### 3.3 Symbol-level rolling features (Week 2, Day 1–2)

- Rolling over `time_id` within day, windows ∈ {5, 20, 60}.
- `shift(1)` before the rolling reduction — same-slot inputs leak.
- Decay-weighted EWM variants, half-lives ∈ {5, 20}.

### 3.4 Target transformations (Week 2, Day 3)

- Winsorize `responder_6` at ±6σ on train only. Evaluate as a training
  target (loss computation), not as a reported metric.

### 3.5 Feature-family ablation (Week 2, Day 4–5)

- For each family (lag refinements, cross-sectional, symbol-rolling,
  target transforms): run a FOLDS_A evaluation with only that family
  added to the Phase 2 best baseline.
- Report Δ(R²) and Δ(non-lag R²) per family, with std across folds.
- A family advances to Phase 4 only if it improves FOLDS_A mean R² by
  `0.5 · σ_min` or improves non-lag R² by `ε_nl`.

### 3.6 LB check (1 submission, ~end of Phase 3)

- Submit the best-feature-engineered model + simple ridge to Kaggle.
- Record LB. Compare to FOLDS_A CV. **If `|CV − LB| > 2 · σ_min`,
  pause and investigate** before Phase 4.
- Budget cost: 1/10 (cumulative 2/10).

### Phase 3 Gate

- At least one non-lag feature family advances.
- CV↔LB gap at the Phase 3 submission is within `2 · σ_min`, or a
  documented explanation for the divergence exists.

---

## Phase 4 — Model Architecture (2 weeks)

Goal: pick one architecture. All decisions on FOLDS_A.

### 4.1 Candidates

- MLP on selected features + lag residual head.
- GRU per (symbol, day) sequence, with lag-residual init.
- GRU with lag-dropout `p ∈ {0.0, 0.15, 0.3}` to force feature learning.
- (Conditional) Short-sequence Transformer over intraday slots — only
  evaluated if Phase 3 cross-sectional features advanced.
- Two-tower (lag tower + feature tower, combined at output) — only
  evaluated if Phase 3 showed ≥ ε_nl non-lag gain.

### 4.2 Training protocol (inherited from 04-13, correct as written)

- Standardize on each fold's `train_range` only; apply to val.
- AdamW, cosine LR, grad clip 1.0.
- **Early stopping on fold's inner-val only** (per `inner_split()`).
  The fold's `val_range` is never observed during training.
- Seed ensemble size 5, fixed seeds {42, 123, 456, 789, 2024}. Seeds
  are predeclared, not hyperparameters.

### 4.3 Hyperparameter search (FOLDS_A only)

- ≤ 16 configs per architecture. Each trained on all 4 FOLDS_A folds.
- Rank by `mean(FOLDS_A R²) − 0.5 · std(FOLDS_A)` = `cv_score`.
- FOLDS_B is not read. Period.

### 4.4 Non-lag floor requirement

- For the selected architecture, report R² under three modes per fold:
  full features, lag-zeroed, lag-noise-replaced.
- **Acceptance**: FOLDS_A non-lag R² ≥ `NLF_best3 + ε_nl` (from §2.4).
- A candidate that fails is logged and cannot be the final design. If
  no candidate passes, Kill Criterion K2 triggers.

### 4.5 LB check (1 submission, ~end of Phase 4)

- Submit the chosen architecture + its Phase 3 feature set.
- Budget cost: 1/10 (cumulative 3/10).

### Phase 4 Gate

- Chosen architecture's FOLDS_A `cv_score` beats best Phase 2
  baseline by `0.5 · σ_min`.
- Non-lag floor passed (§4.4).
- CV↔LB gap within `2 · σ_min`.

---

## Phase 5 — Ensemble and Calibration (1 week, conditional)

**Skip condition**: if `NLF_best3 ≤ 0.01` and the Phase 4 non-lag R² is
similarly near-zero, ensembling and calibration cannot materially help
— they add variance without addressing the signal ceiling. Go straight
to Phase 6 with the frozen Phase 4 design.

### 5.1 Seed ensemble

- 5 seeds, predeclared. Equal weights. No per-seed tuning.

### 5.2 Regime conditioning (conditional, only if Phase 4 candidate is GRU/Transformer)

- Regime state from **causal** rolling volatility of responder_6,
  `shift(1)` applied.
- Thresholds fit per FOLDS_A fold on train only; frozen on val.
- Audit script fixture: two rows at same `date_id` but different
  `time_id` must produce different regime features (catches
  time_id/date_id swap).

### 5.3 Cross-architecture blend (conditional, only if 2+ architectures passed §4.4)

- Fit blend weights on FOLDS_A OOF predictions.
- Keep blend only if it beats the best single architecture by
  `0.5 · σ_min` on FOLDS_A. Else drop; seed ensemble only.

### 5.4 Shrinkage (conditional)

- Uncertainty-weighted shrinkage: per prediction, estimate variance
  across seeds; shrink high-variance predictions toward zero by
  `γ(var)`.
- Fit γ on FOLDS_A folds 1–2, evaluate on folds 3–4. Keep only if
  the held-out half improves; else drop.

### Phase 5 Gate

- Whatever is kept (seed ensemble ± regime ± blend ± shrinkage)
  beats Phase 4 single-model on FOLDS_A by `0.5 · σ_min`, or the
  component is dropped.

---

## Phase 6 — Outer Evaluation and Holdout Run (1 week)

Goal: produce the final number(s). One FOLDS_B read, one HOLDOUT read,
one LB submission.

### 6.1 FOLDS_B read (one-shot)

- Preconditions: all prior gates passed; design (features,
  architecture, hyperparameters, ensemble, regime, shrinkage) frozen
  in `config/final.yaml`; audit script clean.
- Set `JS_PHASE_UNLOCK=FOLDS_B_PHASE6`. Evaluate the frozen design on
  FOLDS_B's one fold (train=(0, 1300), val=(1301, 1443)). Record.
- **No design change may follow this step.** If the FOLDS_B score
  triggers Kill Criterion K3 (FOLDS_B R² < FOLDS_A mean R² − 2·σ_min),
  the cycle proceeds to holdout and submission anyway, with the
  failure noted in the writeup. "Exit on fail" is not the rule.

### 6.2 Build `holdout_gate.yaml`

- `src/build_holdout_gate.py` reads FOLDS_B R², `NLF_best3`, `ε_nl`,
  `SIGMA_MIN`, and produces:
  ```yaml
  full_feature_relative:  holdout R² ≥ FOLDS_B_R2 - 2 * SIGMA_MIN
  full_feature_absolute:  holdout R² ≥ 0.5 * FOLDS_B_R2
  non_lag:                holdout non-lag R² ≥ NLF_best3 + ε_nl
  stability:              no 50-day window has R² < -SIGMA_MIN
  ```
- Commit this file before holdout unlock. Re-run the builder on PR;
  any mismatch with committed file blocks the PR.

### 6.3 Holdout run (one-shot)

- Retrain frozen design on dates 0–1443 (train + val merged).
- Set `JS_PHASE_UNLOCK=HOLDOUT_PHASE6`. Predict on dates 1444–1571.
  RESERVE (1572–1698) stays locked (honor system).
- Report: full-feature R², non-lag R², per-50-day-window R².
- Evaluate against `holdout_gate.yaml`.

### 6.4 Final LB submission

- Submit frozen design retrained on 0–1443. Record LB.
- Budget cost: up to 5/10 remaining. Typical usage 2–3 for final
  seed/architecture variants.

### Phase 6 Gate

- `holdout_gate.yaml` present, committed, matches builder output.
- Holdout reported (pass or fail); writeup records which.
- LB submission recorded.

### If the holdout gate fails

- Record CV (FOLDS_A + FOLDS_B) vs holdout vs LB in the writeup.
- **Ship anyway**: the frozen design goes up as the competition
  submission. The failure is the cycle's finding, not its blocker.
- Do not propose fixes in this writeup; fixes are a next-cycle
  problem.

---

## The Non-Lag Floor, Consolidated

The **sole** non-lag gate used everywhere in this plan:

```
non-lag R² ≥ NLF_best3 + ε_nl
```

where:
- `NLF_best3` = `max over {ridge, lgbm, mlp}` of `mean(FOLDS_A non-lag R²) − 0.5 · std(FOLDS_A)`
- `ε_nl` = `std(NLF_best3 across FOLDS_A folds, using the winning model family)`

Both computed once in Phase 2.4 and frozen. Referenced by §4.4 (architecture
acceptance) and §6.2 (holdout gate). Legacy `0.01` constant from 04-13 is
deleted everywhere.

---

## The FOLDS_A / FOLDS_B Discipline, Corrected

| Decision                                    | Pool used     |
|---------------------------------------------|---------------|
| Hyperparameters (all phases)                | FOLDS_A only  |
| Architecture choice                         | FOLDS_A only  |
| Feature-family keep/drop                    | FOLDS_A only  |
| Ensemble component keep/drop                | FOLDS_A only  |
| Shrinkage keep/drop                         | FOLDS_A only  |
| Early stopping within a fold                | fold's inner-val only |
| Cross-fold feature aggregation applied to…  | FOLDS_B or HOLDOUT only — never FOLDS_A |
| FOLDS_B read                                | once, at §6.1 |
| HOLDOUT read                                | once, at §6.3 |
| RESERVE read                                | **not at all** this cycle |
| LB submissions                              | ≤ 10 total (§0.5 budget) |

Enforcement:
- `load_folds_b_val()` / `load_holdout()` / `load_reserve()` raise
  unless `JS_PHASE_UNLOCK` is set (§0.3).
- `lb_tracker.md` is grep'd on every PR; submission count must not
  exceed the budget.
- Leakage audit (§0.4) enforces the three concrete leakage classes.

---

## Definition of Ship

A design ships when **all** hold:

1. **FOLDS_A `cv_score`** beats the freshly-measured `k · lag1_r6`
   baseline by `0.5 · σ_min`.
2. **FOLDS_A non-lag R²** ≥ `NLF_best3 + ε_nl`.
3. **FOLDS_B** read exactly once at §6.1; `lb_tracker.md` within
   budget; `JS_PHASE_UNLOCK` usage logged.
4. **No FOLDS_B-derived design decision**; audit script clean; rules
   from §2.5 (fold-local selection) respected.
5. **`holdout_gate.yaml`** committed before holdout unlock; generated
   by builder script; matches on re-run.
6. **Holdout run completed**; result recorded (pass or fail).
7. **LB submission made**; CV↔LB gap within `2 · σ_min` OR the
   discrepancy is analyzed in the writeup.

"Ship" in this plan means "submit to Kaggle with honest writeup."
A holdout failure still ships; the writeup records the failure. The
alternative — not submitting — is strictly worse in a competition.

---

## Kill Criteria (softened from 04-13)

These trigger **reduced ambition**, not cycle exit:

- **K1 (Phase 2)**: best tree baseline fails to beat `k · lag1_r6` by
  `0.5 · σ_min`. Interpretation: features contain no tree-extractable
  signal beyond lag. Response: proceed, but expect Phase 4 candidates
  to reduce to "better-regularized lag predictors" and document that
  up front.
- **K2 (Phase 4)**: no architecture passes the non-lag floor.
  Response: ship the best lag-only design with honest framing.
  Submit to LB; the competition may reward that anyway.
- **K3 (Phase 6.1)**: FOLDS_B R² < FOLDS_A mean − 2·σ_min.
  Response: **do not retune**. Ship the frozen design; record the
  Stage A↔B gap; the writeup is the finding.
- **K4 (any phase)**: CV↔LB divergence > `2 · σ_min` at two
  consecutive submissions. Response: pause modeling; investigate
  scoring/pipeline/data before spending more budget.
- **K5 (Phase 0 drags on)**: leakage audit keeps finding new classes
  after two full fix cycles. Response: the abstraction boundary is
  wrong; restart `data.py` from scratch.

Exit is reserved for K5. Everything else produces a submission and
a writeup.

---

## Timeline

| Week | Phase                                           | Cum. LB budget |
|------|-------------------------------------------------|----------------|
| 1    | Phase 0 infra + Phase 1 diagnostics (Days 1–3 Phase 0, Days 4–7 Phase 1.1–1.2) | 1/10 |
| 2    | Phase 1 continues (1.3 EDA, 1.4 microstructure) | 1/10           |
| 3    | Phase 2 baselines + freeze SIGMA_MIN            | 1/10           |
| 4–5  | Phase 3 feature engineering                     | 2/10           |
| 6–7  | Phase 4 architecture                            | 3/10           |
| 8    | Phase 5 ensemble/calibration (conditional)      | 3/10           |
| 9    | Phase 6 FOLDS_B + holdout + final LB            | 6–8/10         |
| 10   | Writeup, tidying, reserve submission slots      | 8–10/10        |

Phase 0 is 3 days; Phase 1 diagnostics start Day 4 of Week 1. This
front-loads the cheap high-EV work (scoring verification, lag
mechanism, LB baseline) into the first week where it reframes the
rest of the project.

---

## Continuous Protocol

- Every script output contains a `LoadManifest` (date range, row
  count, sha256). Reports without manifests rejected at PR.
- Every model/feature/post-hoc step reports its non-lag R².
- `lb_tracker.md` updated after every Kaggle submission.
- Leakage audit runs in CI on every push touching `src/`.
- `JS_PHASE_UNLOCK` usage logged with timestamp in `reports/unlocks.log`.
- If a rule in this plan turns out to cost more than it saves, edit
  the plan. Protocol serves science, not vice versa.

---

## Summary of Changes vs 04-13

| Area                        | 04-13                                    | 04-17                                      |
|-----------------------------|------------------------------------------|--------------------------------------------|
| Primary ground truth        | Local HOLDOUT (1444–1571)                | Kaggle LB                                  |
| Phase 0 duration            | 2 weeks                                  | 3 days                                     |
| Leakage audit               | ~10 checks, fixtures, provenance chains  | 3 concrete checks, 50 LoC                  |
| Sealed storage              | `chmod 700`, phase tokens, external log  | Honor system, `JS_PHASE_UNLOCK` env var    |
| Non-lag reference           | `NLF_A_ref` (LightGBM only)              | `NLF_best3` (max over 3 model families)    |
| σ_min                       | Predeclared `0.02` constant              | Derived from FOLDS_A R² variance at Phase 2 |
| On holdout failure          | Exit cycle, no ship                      | Ship with writeup                          |
| Cross-fold feature agg      | Blanket ban                              | Allowed; apply to B/HOLDOUT only, not A    |
| LB submissions              | Not mentioned                            | Budget of 10, tracked in `lb_tracker.md`   |
| Microstructure investigation | Absent                                  | Phase 1.4, first-class                     |
| Scoring formula check       | Assumed correct                          | Day 1, byte-for-byte against Kaggle metric |
| lag1_r6 mechanism           | Assumed leakage                          | Investigated Phase 1.2 before rebuild      |

The 04-13 plan's central instincts (CV discipline, non-lag floor,
FOLDS_A/B separation, one-shot holdout, audit script) are preserved.
What changed is the allocation of effort and the admission that on a
single-user workstation competing on Kaggle, the real out-of-sample
signal is the LB and the real enforcement mechanism is a submission
budget, not filesystem permissions.
