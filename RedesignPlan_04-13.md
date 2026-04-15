# Project Redesign — 2026-04-13

Full redesign of the Jane Street 2024 project. The previous execution
(Phases 1–9 in `previousResultProblem.md`) produced numbers that are not
trustworthy as out-of-sample evidence, for four independently sufficient
reasons:

1. Model selection was done on a single validation slice (1189–1443),
   turning the validation set into part of the training loop.
2. The regime module leaked future statistics into validation labels.
3. The regime code silently used `date_id` where `time_id` was intended,
   fabricating spurious intraday signal.
4. ~53% of the dataset (dates 0–899) was never used for training; the
   900 cutoff was never justified.

In addition, the headline Val R² ≈ 0.88 vs. competition leaderboard ≈ 0.014
(a 60× gap) strongly suggests our approach rides a single feature
(`lag1_r6`, autocorr ≈ 0.90) that does not generalize to live market data.

This redesign rebuilds the project around **protocol discipline first,
modeling second**. Every phase below has explicit gates. Anything that
cannot pass its gate does not move forward.

---

## Guiding Principles

1. **Leakage-clean by construction.** Every feature, threshold, and
   statistic is fit on train only and applied to val/holdout with frozen
   parameters. Enforced by an audit script that must exit clean before
   any result is accepted.
2. **Walk-forward CV, not single-slice.** All model selection and
   hyperparameter decisions go through expanding-window CV. Score =
   `mean(R²) − 0.5 · std(R²)` across folds.
3. **Non-lag floor is a first-class metric.** Every model reports R²
   with lag features (a) intact, (b) zeroed, (c) noise-replaced. Ship
   criterion includes a minimum non-lag R².
4. **Full data range.** Training uses dates 0–1188. No arbitrary cutoffs.
5. **Holdout is sacred.** Dates 1444–1698 are touched once at the end.
6. **Reject-on-failure, not patch-on-failure.** If a phase gate fails,
   the phase is re-done, not worked around.

---

## Phase 0 — Research Infrastructure (new, blocking all modeling)

Goal: set up the scaffolding that every later phase assumes exists. No
modeling happens until Phase 0 is complete.

### 0.1 Canonical splits
- `src/splits.py`: single source of truth for date ranges. Exports
  `TRAIN_RANGE = (0, 1188)`, `VAL_RANGE = (1189, 1443)`,
  `HOLDOUT_RANGE = (1444, 1571)` (report-only),
  `RESERVE_RANGE = (1572, 1698)` (sealed, untouched until a future
  research cycle). Delete all hard-coded `900, 1188`.
- The holdout split into two halves is deliberate: Codex review flagged
  that gating ship on a single one-shot holdout silently turns it into
  another development signal once it surprises us. RESERVE is sealed
  and its file is permission-locked at the OS level; it does not exist
  for this research cycle.
- Walk-forward folds defined here and imported, not re-declared.
  Two roles, strictly disjoint at the label level. Codex review
  flagged that the prior split used FOLDS_B at every phase gate
  (§4, §5, §6) to decide whether regime/ensemble/shrinkage stay;
  that turns FOLDS_B into a repeatedly-read selection pool rather
  than a one-shot outer eval. It also placed FOLDS_B earliest fold
  at date 901 — mid-range, not close to holdout — which diluted the
  signal most relevant to generalization. Fix:
  ```
  FOLDS_A = [   # TUNING + PHASE-GATE DECISIONS: hyperparameters,
                # architecture, blend weights, shrinkage, non-lag
                # baseline calibration, phase-gate go/no-go. Every
                # pre-holdout gate in this plan reads FOLDS_A only.
      (train=(0, 500),   val=(501, 700)),
      (train=(0, 700),   val=(701, 900)),
      (train=(0, 900),   val=(901, 1100)),
      (train=(0, 1100),  val=(1101, 1300)),
  ]
  FOLDS_B = [   # OUTER EVAL, ONE-SHOT. Read exactly once at the
                # Phase 8 precondition check, after the full design
                # (features, architecture, hyperparameters, ensemble,
                # regime, calibration) is frozen on FOLDS_A. Latest
                # range available pre-holdout, so it is the closest
                # distributional proxy to HOLDOUT.
      (train=(0, 1300),  val=(1301, 1443)),
  ]
  # Each fold also carves its own INNER-VAL out of the tail of its own
  # train_range (last ~10% of train dates) for early stopping and any
  # training-time decision. Inner-val is NEVER the fold's val_range.
  def inner_split(train_lo, train_hi, frac=0.10):
      cut = train_hi - int(frac * (train_hi - train_lo))
      return (train_lo, cut), (cut + 1, train_hi)
  ```
- **Role separation is enforced, not advisory.** The audit script
  rejects any pipeline that (a) reads a config or parameter derived
  from FOLDS_B val labels into training or design decisions,
  (b) reads FOLDS_B val labels more than once across the full cycle
  (tracked via an append-only `folds_b_read_log` outside the
  workspace — a second read is a release-blocking violation),
  (c) uses a fold's own `val_range` as its early-stopping signal.

### 0.2 Leakage audit script
- `src/audit_leakage.py`: runs the following checks on any pipeline
  module it is pointed at. Exits non-zero on violation. **Fail-closed
  design:** Codex review flagged that sampled audits and
  self-reported provenance markers can pass while still missing the
  leakage classes the protocol depends on catching. The script
  therefore treats missing/invalid provenance as a violation, not a
  warning, and exhaustively covers boundary slots rather than
  sampling.
  - Normalization stats fit on `date_id < val_start`.
  - Any `quantile()` / `percentile()` / threshold fit on train only.
  - Rolling/expanding features: recomputation from strictly-past data
    reproduces the pipeline output to within 1e-9 on **all** val rows
    at day/slot boundaries (first and last `time_id` of every val
    `date_id`, plus the first `date_id` of each val range) and on a
    random 5% of remaining val rows. Day/slot boundaries are where
    look-ahead bugs hide; sampling alone can miss them.
  - Lag responders at `(date_id=d, time_id=t)` equal the previous slot's
    responders — never the same-slot responder.
  - `time_id` and `date_id` are passed in the correct positional order
    (catches C3).
  - **Independent provenance.** Provenance markers are generated by a
    separate `src/provenance.py` module that wraps any data read and
    records the date-range accessed. Artifacts whose provenance is
    emitted by the pipeline code itself (not `src/provenance.py`) are
    rejected. Any artifact missing a provenance record is rejected;
    "fail open on unknown provenance" is not an option.
  - Feature selection artifacts (top-K lists) carry an independent
    provenance marker recording the date range they were fit on;
    reject if it overlaps val.
  - **Training loops forbid val-range peeks.** Any per-epoch metric
    computed on a fold's `val_range` (loss, R², anything) is a
    violation. Early stopping must read from the fold's inner-val only.
  - **Cross-pool leakage.** Any artifact (epoch count, hyperparameter,
    blend weight, shrinkage curve, feature list) loaded by a FOLDS_B
    evaluation run must carry an independent provenance marker
    showing it was fit without reading FOLDS_B val labels. The audit
    script walks the provenance chain and rejects any chain that
    passes through a FOLDS_B val-label consumer, and rejects any
    chain with a missing link.
  - **FOLDS_B read ledger.** An append-only file outside the
    workspace (`/var/kaggle-js/folds_b_read_log`) records every read
    of FOLDS_B val labels. The audit script reads this ledger; more
    than one entry inside a research cycle is a release-blocking
    violation. The ledger is written by the data loader wrapper, not
    by the pipeline being audited.
  - **Fold-local feature selection.** No artifact with provenance
    spanning more than one fold's training window may be loaded as a
    feature list during training. "Stable across ≥ k of N folds"
    aggregations are rejected categorically.
  - **Deterministic leakage fixtures.** The audit script includes a
    test suite of synthetic cases (known look-ahead bugs, off-by-one
    at day boundaries, swapped `time_id`/`date_id`) that the pipeline
    is run against; each must trigger the expected rejection. A
    pipeline release is blocked if any fixture regresses.

### 0.3 Metric and scoring library
- `src/metrics.py`:
  - `weighted_r2(y, yhat, w)` — sample-weighted zero-mean R².
  - `cv_score(fold_results, lam=0.5)` — returns `mean − lam · std`.
  - `non_lag_r2(model, X, y, w, lag_cols, mode={"zero","noise"})`.

### 0.4 Data loader with provenance
- `src/data.py` (rewrite): every `load_pass()` returns a
  `LoadManifest` recording the date range, row count, columns, and a
  hash of the loaded rows. Manifests are saved alongside every report.
  This makes silent data drops (like the 900 cutoff) impossible to hide.

### Phase 0 Gate
- Audit script runs clean against `data.py`.
- CV folds produce identical indices across reruns.
- Metric library has unit tests with known inputs.

---

## Phase 1 — Data Integrity and Full-Range EDA (redo)

Goal: rebuild EDA on the **full** training range (0–1188), not the
truncated 900–1188. Deliver the facts needed to justify every design
choice downstream.

### 1.1 Data re-verification
- Re-verify all 10 partitions (no truncation), 47.1M rows, 92 columns.
- Confirm no hidden date gaps in 0–1188.

### 1.2 Distribution audit over 0–1188
- Responder_6: mean, std, skew, kurtosis per 100-day block. Document
  drift.
- Weight distribution per block.
- Feature null rates per block. Investigate any feature with rate > 10%.
- Decision: which features, if any, have degenerate behavior on 0–899
  that justified the original cutoff? If none, the cutoff was arbitrary.

### 1.3 Temporal structure
- Responder_6 rolling std per 50-day window across 0–1698.
- Autocorrelation of responder_6 at lag 1, 5, 20 across the full range
  to check whether `lag1_r6` corr ≈ 0.90 holds on 0–899 as well as on
  900–1188. (Phase 8.4 fold-1 result suggests yes; verify explicitly.)
- Per-symbol activity: rows per (symbol, date_id) to check whether
  symbols enter/leave the universe.

### 1.4 Feature-target relationships (descriptive only — do NOT drive CV feature sets)
- Any statistic computed here is for human understanding of the data,
  not for deciding which features go into training.
- **Per-fold feature decisions happen inside CV.** Computing a single
  "selected feature list" on dates 0–1100 (or any global range) and
  then reusing it across CV folds leaks future information into early
  folds: Fold 1's training ends at date 500, but its features would
  have been chosen using data through 1100. This is a direct look-ahead.
- Sign stability and redundancy clustering, if computed at all, must
  be tagged `descriptive_only=True` in the manifest and the audit
  script rejects any CV pipeline that reads from them.
- The **only** acceptable use of global statistics at this phase is to
  rule out features with fundamental data issues (e.g., all-null
  columns for more than 50% of all dates). That is a data-quality
  decision, not a feature-selection decision.

### 1.5 Lag file contract
- Re-derive the exact causal rule for lag responders from the provided
  lags file template. Document with timing diagram. Add a test case to
  the audit script.

### Phase 1 Gate
- All EDA notebooks/scripts report the date range used in their
  manifests.
- Either (a) justified exclusion of some early dates with a specific
  finding, or (b) confirmation that dates 0–899 are usable.

---

## Phase 2 — Honest Baselines (redo)

Goal: produce reference numbers under the new protocol that all later
work must beat.

### 2.1 Dumb baselines (walk-forward)
- Predict zero → R² = 0 by construction, no compute needed.
- Predict `k · lag1_r6` with `k ∈ {0.85, 0.90, 0.95, 1.00}`. Fit `k` on
  each fold's train; evaluate on fold's val. Report mean / std.

### 2.2 Linear baselines (walk-forward)
- Ridge on full 79 features + lag responders.
- Ridge on residual of `0.90 · lag1_r6`.
- Report CV score and non-lag R².

### 2.3 Tree baselines (walk-forward)
- LightGBM and XGBoost, same hyperparameters across folds.
- Report CV score, non-lag R², and feature importances per fold.
- Per-fold importance tables are **descriptive only** — they may
  surface a hypothesis like "feature X looks regime-specific," but
  that hypothesis cannot be acted on as a training decision in this
  cycle. Acting on it would smuggle cross-fold information into the
  feature set. Write it down and move on.

### 2.4 Feature selection (fully fold-local, no global "stable set")
Codex review flagged that a "stable across ≥ 4 of 5 folds" filter
silently aggregates signal from folds outside the scored evaluation —
including Stage A folds — and smuggles cross-fold information into the
Stage B score. That reintroduces exactly the leakage we are trying to
eliminate. Rule:

- **No global feature set is ever built.** Each fold picks its own
  features from its own training data only. Fold 3's LightGBM-gain
  selection knows nothing about folds 1, 2, 4, or 5.
- Inside a fold: fit LightGBM on `train_range`, keep features to 90%
  cumulative gain, retrain the fold's model on the fold-local selected
  set, score on the fold's `val_range`.
- Feature-importance comparisons across folds in Phase 2.3 are
  **descriptive only**. They may be printed for human intuition; the
  audit script rejects any pipeline that reads them into a training
  decision.
- No "top-K by cross-fold frequency" lists. No "majority-vote"
  selection. No Phase-1-style global stability ranks. These all
  constitute cross-fold aggregation of feature-selection signal and
  are disallowed.
- For the holdout run in Phase 8, feature selection is done once on
  the full `TRAIN_RANGE ∪ VAL_RANGE` (0–1443) training window using
  the same procedure — still fold-local semantics (one window, one
  selection), not a cross-fold vote.

### Phase 2 Gate
- Non-lag R² for the best tree baseline is reported.
- Beating the walk-forward `k · lag1_r6` baseline by `0.5 · std` is the
  new reference.

---

## Phase 3 — Feature Engineering (redo, leakage-clean)

Goal: revisit feature engineering under the audit, without the previous
"no improvement" verdicts biasing the search, since those were measured
on one slice with mixed hygiene.

### 3.1 Market-level features
- Per `(date_id, time_id)` aggregates: mean, std, cross-sectional rank
  of top stable features.
- Every aggregate shifted by at least 1 `time_id` within day to prevent
  same-slot leakage (responders of other symbols at the same `time_id`
  can leak the target).
- Verify with audit script.

### 3.2 Symbol rolling features
- Rolling over `time_id` within day, window ∈ {5, 20, 60}.
- `shift(1)` before the rolling reduction.
- Decay-weighted variants (EWM) with half-life ∈ {5, 20}.

### 3.3 Lag responders
- Canonical lag responders (lag1_r0…r8) are the primary signal carrier.
- **New:** multi-horizon lag aggregates: mean of lag1 across the last
  `k` intraday slots for `k ∈ {3, 5, 10}`. Tests whether the single-lag
  signal can be smoothed for robustness.

### 3.4 Target transformations (train-only)
- Winsorize responder_6 at ±6σ on train only. Evaluate whether it
  improves CV score. Apply only at loss computation, not to reported R².

### Phase 3 Gate
- Each feature family evaluated as a delta over the Phase 2 best
  baseline under walk-forward.
- Any feature that needs val data to compute is rejected outright.

---

## Phase 4 — Model Architecture (redo, with non-lag floor)

Goal: compare architectures fairly under CV, and insist each reports its
non-lag floor.

### 4.1 Candidates
- MLP + lag residual (reference point).
- GRU per (symbol, day) sequences, with lag residual init.
- GRU with lag-dropout `p ∈ {0.0, 0.15, 0.3}` to force feature learning.
- **New:** feature-primary Transformer (short-sequence attention over
  recent intraday slots). Hypothesis: attention may extract weak
  cross-feature structure that GRU misses.
- **New:** two-tower model: one tower sees only lag responders, the
  other sees only features; combined at the output. Allows measuring
  each tower's marginal contribution directly.

### 4.2 Training protocol
- Standardize on each fold's `train_range` only; applied to val; full
  `LoadManifest` recorded.
- AdamW, cosine LR schedule, grad clip 1.0.
- **Early stopping uses the fold's inner-val only** (last ~10% of its
  own `train_range`, per `inner_split()` in Phase 0). The fold's
  `val_range` is never observed during training for any purpose —
  loss, metric, stopping, LR schedule, nothing. This closes the
  "peeking at Stage B val labels via early stop" path.
- **"CV-mean early stopping" is explicitly banned.** Stopping on a
  signal that aggregates FOLDS_B val labels is the same as tuning on
  FOLDS_B, just at the training-loop level. The audit script rejects
  any training loop that computes a metric on `val_range` per epoch.
- Seed-ensemble size 5, fixed seeds {42, 123, 456, 789, 2024}. Seeds
  are predeclared; they are not hyperparameters.

### 4.3 Hyperparameter search (all decisions on FOLDS_A; FOLDS_B is untouched until Phase 8)
Codex review pushed back on any scheme where FOLDS_B influences
training-time decisions **or** phase-gate decisions. An earlier draft
read FOLDS_B at the Phase 4/5/6 gates and called that "reporting" —
but a gate that can drop regime/ensemble/calibration based on FOLDS_B
is selection on FOLDS_B, just once per phase instead of once per
epoch. The revised rule is the strongest version of the disjoint
split:

- **All hyperparameter choices, architecture choices, and epoch
  decisions are made using FOLDS_A only.** This includes depth,
  width, dropout, learning rate, weight decay, chunk size, ensemble
  size, and any early-stopping epoch cap.
- Within each FOLDS_A fold, training uses that fold's inner-val for
  loss curves and stopping. FOLDS_A `val_range` may be inspected post
  hoc for ranking configurations, but is **not** used during the
  training loop.
- **Reduced grid:** ≤ 16 configs total. Each config is trained on all
  four FOLDS_A folds. Rank by
  `mean(FOLDS_A val R²) − 0.5 · std(FOLDS_A)`. Pick one winner. Four
  folds gives enough variance signal that the ranking is not a
  coin-flip.
- **FOLDS_B is untouched during Phase 4.** The one FOLDS_B fold
  (train=(0, 1300), val=(1301, 1443)) is not read, period. FOLDS_B is
  read exactly once, at the Phase 8 precondition check, after the
  full design is frozen.
- Final config frozen before FOLDS_B is read and before any full
  retrain on (0–1443) for the holdout run.

### 4.4 Non-lag floor requirement (threshold derived, not hard-coded)
Codex flagged that the previous `non-lag R² ≥ 0.01` threshold was
inherited from the old protocol's feature-only LightGBM result, which
is exactly the number the redesign declared untrustworthy. Fix:

- **Establish the non-lag reference freshly under the new protocol.**
  Train a feature-only LightGBM (all 79 features; lag columns
  zero-ablated from input) on FOLDS_A using the same tuning rules as
  §4.3. Call the resulting score `NLF_A_ref` (mean − 0.5·std over
  FOLDS_A). This is the leakage-clean non-lag reference.
- For each candidate architecture, measure non-lag R² under three
  modes on each FOLDS_A fold:
  - Full features
  - Lag features zeroed
  - Lag features replaced with matched-variance noise
- **The acceptance threshold is `NLF_A_ref + ε`**, where ε is
  `std(NLF_A_ref across FOLDS_A folds)`. A candidate advances only if
  its FOLDS_A non-lag R² beats the fresh reference by this
  non-trivial margin. This same margin is carried through to the
  holdout gate (§8.3) — it is not relaxed anywhere.
- A model that fails the non-lag floor is logged but cannot be the
  final design.
- If `NLF_A_ref` itself comes back near zero, that is the expected
  outcome given the competition-leaderboard evidence; the ship
  condition then demands that the model add real non-lag signal over
  that near-zero baseline by the full `+ε` margin, not merely match
  it.

### Phase 4 Gate
- `NLF_A_ref` measured and recorded.
- At least one architecture exceeds the non-lag threshold
  `NLF_A_ref + std(NLF_A_ref)` on FOLDS_A.
- Chosen architecture beats the Phase 2 best tree baseline by
  `0.5 · std` under FOLDS_A CV.
- **FOLDS_B is not consulted at this gate.**

---

## Phase 5 — Regime and Ensemble (rebuild)

Goal: restore regime conditioning only with leakage-clean construction,
and define the ensemble strategy under CV.

### 5.1 Regime module (rebuilt)
- Regime state from rolling volatility of responder_6 over a
  **causal** window (last N intraday slots with `shift(1)`).
- Thresholds fit on train dates only; persisted; frozen on val/holdout.
- Thresholds recomputed per CV fold (train-only) to measure fold
  stability.
- Unit test in audit script: `time_id` and `date_id` are distinct; a
  same-day row pair with different `time_id` must have different
  regime features.

### 5.2 Regime-conditional architecture
- Shared encoder + 3 regime-specific heads, routed by regime label.
- Baseline alternative: a single model that receives regime state as an
  input feature. Whichever wins on CV gets promoted; both report the
  non-lag floor.

### 5.3 Ensembling (all decisions on FOLDS_A)
- Seed ensemble (5 seeds, predeclared). Equal weights; no per-seed
  weighting that would itself be a tuned parameter.
- **Cross-architecture blend weights are fit on FOLDS_A OOF
  predictions and the decision to keep/drop the blend is also made
  on FOLDS_A.** Codex review flagged the earlier "fit on A, report
  on B" framing: if the Phase 5 Gate reads FOLDS_B to decide whether
  to skip regime, FOLDS_B becomes a selection signal. Fix: FOLDS_B
  is not read at Phase 5.
- Keep-or-drop rule: the frozen blend must beat the best single
  architecture by `0.5 · std` on FOLDS_A. Otherwise drop the blend
  entirely and ship a single-architecture seed ensemble.
- Re-measure the GRU+MLP blend under this protocol; the old "MLP
  weight = 0" conclusion was measured on a single slice and may not
  survive clean measurement.

### Phase 5 Gate
- Regime module passes the audit.
- Frozen-weight ensemble's FOLDS_A score beats the Phase 4
  single-architecture best FOLDS_A score by `0.5 · std`. If not, skip
  regime and use the seed ensemble only (no blend fitting at all).
- **FOLDS_B is not consulted at this gate.**

---

## Phase 6 — Post-hoc Calibration (new positioning)

Goal: shrinkage is a policy, not a model. Fit it on CV only.

### 6.1 Prediction shrinkage (all decisions on FOLDS_A)
- Uncertainty-weighted shrinkage: for each prediction, estimate
  variance across ensemble seeds; shrink high-variance predictions
  toward zero by factor `γ(var)`.
- **Fit the `γ` curve on FOLDS_A OOF predictions only, AND decide
  keep/drop on FOLDS_A.** Codex review flagged that any decision
  which reads FOLDS_B is selection on FOLDS_B, even if the decision
  is phrased as "reporting." FOLDS_B stays untouched through Phase 6.
- The keep-or-drop rule: fit `γ` on half of FOLDS_A OOF predictions
  (folds 1–2), measure its effect on the other half (folds 3–4). If
  the held-out half shows no improvement, drop shrinkage.
- If `γ` is degenerate (e.g., collapses to the identity) on FOLDS_A,
  shrinkage is dropped, not re-parameterized.

### 6.2 Weight-aware calibration
- Verify the loss-weight and scoring-weight are applied consistently.
  Check that weight winsorization (if any) is train-only.

### Phase 6 Gate
- Frozen shrinkage policy improves the held-out FOLDS_A half-score
  (or at minimum leaves it unchanged).
- Calibration curve is stable across adjacent FOLDS_A folds.
- **FOLDS_B is not consulted at this gate.**

---

## Phase 7 — Online / Adaptive Component (optional)

Goal: revisit whether online adaptation can help, only now that the
frozen baseline is protocol-clean.

### 7.1 Decision point
- If Phase 4 non-lag floor is below 0.05, online adaptation cannot
  meaningfully improve real live performance — skip this phase.
- If non-lag floor is > 0.05, online adaptation is worth one controlled
  experiment: exponentially-weighted residual correction, no gradient
  updates (which already degraded in the old Phase 6).

### Phase 7 Gate
- CV score improves; otherwise online component is dropped.

---

## Phase 8 — Final Holdout Evaluation (one shot)

Goal: produce a single, honest number.

### 8.1 Preconditions before touching holdout
- All earlier gates passed, with all pre-holdout decisions made on
  FOLDS_A only.
- **FOLDS_B read, exactly once, here.** The frozen design (features,
  architecture, hyperparameters, ensemble, regime, calibration) is
  evaluated on FOLDS_B. Results are recorded. The `folds_b_read_log`
  records this read. No design change may be made after this step —
  if the FOLDS_B score is disappointing, the cycle still proceeds to
  the holdout run, or exits. There is no "re-tune after seeing B."
- Audit script clean on every pipeline module.
- Final configuration file checked in, including: features, model
  architecture, hyperparameters, ensemble size, regime thresholds,
  shrinkage curve. Nothing else may change.

### 8.2 Holdout run
- Train final design on dates 0–1443 (train + val merged, allowed now
  because CV and val have already done their job and the design is
  frozen).
- Predict on dates **1444–1571** only. RESERVE (1572–1698) stays
  sealed.
- Report: (a) full-feature R², (b) non-lag R², (c) per-50-day-window
  R² to show stability across the holdout window.

### 8.3 Holdout is a one-shot release gate with exit-on-fail discipline

The earlier "report-only" framing was itself flagged by Codex: the plan
argues the approach likely overfits a historical lag artifact, yet the
release gates live entirely on Stage B folds (historical data). Without
any gate on truly unseen forward data, the protocol certifies designs
on exactly the data pattern we already suspect is the problem.

Reconcile both concerns — preserve one-shot discipline **and** gate on
unseen data — with the following rules:

- **Single source of truth: `holdout_gate.yaml`.** The thresholds
  below are the **only** place in this plan where holdout pass/fail
  rules are specified. The Definition of Ship section references
  `holdout_gate.yaml` by name instead of restating rules. Codex
  flagged an earlier draft where §8.3 and the ship definition stated
  different non-lag thresholds (`≥ 0.01` vs `≥ NLF_A_ref`), which
  would have let someone choose the easier rule after seeing the
  result. One file, one rule, mechanically enforced.
- **Mechanical derivation.** `holdout_gate.yaml` is produced by a
  script `src/build_holdout_gate.py` that reads the FOLDS_B score
  (single number, from the one-shot Phase 8.1 read), `NLF_A_ref` and
  its std from §4.4, and the predeclared constants from
  `src/splits.py`. The script runs **before** the holdout data is
  unsealed and writes the frozen YAML. The file is committed to git
  at that point. The audit script refuses to authorize Phase 8 until
  `holdout_gate.yaml` is present, committed, and matches the output
  of re-running the builder script (ensuring it was not hand-edited).
- **Predeclared pass conditions** (all must hold):
  1. **Relative full-feature gate:**
     `holdout R² ≥ FOLDS_B R² − 2 · σ_min`
     where `σ_min` is a predeclared historical-variance constant in
     `src/splits.py` (initial proposal: `σ_min = 0.02` based on old
     multi-seed ensemble variance; may be revised after Phase 2 once
     a clean FOLDS_A variance estimate is available, but only before
     FOLDS_B is read). With one FOLDS_B fold we cannot estimate
     `std(FOLDS_B)` from data; using a predeclared constant
     eliminates the "high FOLDS_B variance → permissive gate"
     loophole Codex flagged.
  2. **Absolute full-feature floor:**
     `holdout R² ≥ 0.5 · FOLDS_B R²`
     A model that drops to half of the FOLDS_B score fails regardless
     of the relative gate.
  3. **Non-lag gate:**
     `holdout non-lag R² ≥ NLF_A_ref + std(NLF_A_ref)`.
     Codex review flagged that dropping the `+ std` margin here
     (while keeping it in the pre-holdout §4.4 acceptance test) would
     allow a predominantly lag-dependent model to ship as long as the
     lag artifact survives one more slice — exactly the failure mode
     the redesign is supposed to guard against. The holdout non-lag
     gate therefore carries the **same** margin used to accept the
     architecture in §4.4; it is never relaxed. `NLF_A_ref` and its
     std are copied verbatim from the §4.4 artifact into
     `holdout_gate.yaml`. The legacy `0.01` constant is deleted
     everywhere.
  4. **Stability:** No per-50-day-window holdout R² below zero.
- **One-shot.** The gate fires exactly once, on `HOLDOUT_RANGE
  (1444, 1571)`. There is no second attempt inside this research cycle.
- **Exit-on-fail, not iterate-on-fail.** If any gate threshold is
  missed, the cycle's ship decision is **no-ship**, and the team
  enters a cooling-off period. No model, feature, or hyperparameter
  change may be justified by "the holdout was X." The failure writeup
  records the CV-to-holdout gap as the finding.
- **RESERVE is the next-cycle gate.** Any future cycle that wants to
  iterate post-failure must (a) wait at least one cooling-off period,
  (b) design and freeze the next approach using only CV,
  (c) evaluate the new design on `RESERVE_RANGE (1572, 1698)`, which
  lives in **sealed storage** (see §8.5) until then. Once RESERVE is
  read, it is burned the same way HOLDOUT is burned here.

### 8.5 Sealed storage for HOLDOUT and RESERVE (real enforcement)

Codex flagged that "read-only in the workspace" does not actually
prevent reads, only writes — the supposedly sacred slices were fully
inspectable at any time, which defeats the one-shot claim. Fix with
real access control, not advisory wording.

- **Physical location.** HOLDOUT and RESERVE parquet rows live
  **outside the repo workspace**, in `/var/kaggle-js/sealed/` (or the
  user's platform equivalent). The directory is owned by a separate
  user account with mode `700`; the research account has no read
  access by default.
- **Access is mediated by a single unlock script** `unlock_sealed.py`
  that (a) takes a phase token (`HOLDOUT_PHASE8` or
  `RESERVE_NEXT_CYCLE`), (b) validates the gate preconditions
  (manifest hashes, frozen config, `holdout_gate.yaml` committed,
  audit-script pass token present), (c) appends an immutable log
  entry to a separate append-only file outside the workspace, and
  (d) exposes the slice as a temporary mount or a one-shot stream to
  the evaluation process.
- **The repo contains only schema stubs** (column types, row counts,
  date boundaries) — never the actual HOLDOUT/RESERVE parquet rows.
  `git status` and `ls -R` inside the workspace never surface them.
- **If the access-control mechanism cannot be implemented** (single
  developer workstation, no root privileges, etc.), the honest
  alternative is to **drop the "sealed" claim** and state explicitly
  in the writeup that HOLDOUT/RESERVE integrity rests on researcher
  discipline. In that case the final writeup must record "no
  technical enforcement; honor system" so the ship decision is not
  presented with stronger guarantees than the infrastructure
  actually provides.
- The training data (dates 0–1443) remains in the repo workspace
  unchanged.
- **Why this threads the needle.** A report-only holdout leaves no
  forward-data check and lets us ship something we already suspect
  will fail. An iterative holdout contaminates itself after the first
  surprise. A one-shot gate with exit-on-fail preserves both: the
  holdout is genuinely unseen at decision time, and a failure cannot
  become training signal because the cycle is over.

### 8.4 If the holdout gate fails
- Write up the Stage B CV numbers, the holdout numbers, and the gap.
- Do not propose fixes in the same document. Fixes belong to the next
  cycle, which must use RESERVE for its gate, not HOLDOUT.
- Record the failure as the cycle's deliverable. A clean negative
  result documented under this protocol is more valuable than a
  historically-overfit 0.88.

---

## Continuous Protocol (applies throughout)

- Every script output contains the date-range manifest, the fold
  configuration, and an audit-script pass token. Reports without these
  are rejected at PR time.
- No new feature, model, or post-hoc step is accepted without its
  non-lag R² reported.
- HOLDOUT and RESERVE live in sealed storage outside the repo
  workspace (see §8.5). The workspace contains only schema stubs;
  the actual rows are not readable by the research account until
  `unlock_sealed.py` is invoked with a valid phase token. If sealed
  storage cannot be implemented, the writeup records "honor system,
  no technical enforcement" — the claim is never stronger than the
  mechanism.

---

## Timeline (rough, calendar-weeks)

- W1: Phase 0 infra, incl. leakage audit and CV framework.
- W2: Phase 1 EDA over full range; justify (or remove) the 900 cutoff.
- W3: Phase 2 baselines; non-lag floor measured.
- W4–W5: Phase 3 feature engineering, leakage-clean.
- W6–W7: Phase 4 model architecture + hyperparameter search.
- W8: Phase 5 regime + ensemble rebuild.
- W9: Phase 6 calibration; Phase 7 online decision.
- W10: Phase 8 holdout, writeup.

---

## Kill Criteria

If any of the following hold, the project exits with a negative
result instead of continuing to chase a headline number:

- **Non-lag floor not cleared at Phase 4.** No architecture's FOLDS_A
  non-lag R² exceeds `NLF_A_ref + std(NLF_A_ref)` (the freshly
  measured non-lag reference from §4.4). Interpretation: features
  contain no independent signal beyond what a plain feature-only
  baseline already extracts.
- **FOLDS_A std > 0.02** for the best model. Interpretation: apparent
  skill is regime-specific and will not generalize; do not proceed to
  read FOLDS_B.
- **FOLDS_A → FOLDS_B disagreement at §8.1.** The winning config's
  FOLDS_B R² falls below FOLDS_A mean R² by more than
  `2 · std(FOLDS_A)`. Interpretation: tuning on FOLDS_A did not
  produce a stable choice across later data; do not re-tune, do not
  unseal HOLDOUT, exit.
- **Audit script keeps surfacing new leakage classes** after two full
  fix cycles. Interpretation: the abstraction boundaries are wrong;
  restart `data.py` from scratch before continuing.

A clean negative result under this protocol is more valuable than a
questionable 0.88 under the old one.

---

## Definition of Ship (final)

A design ships when **all** of the following hold.

**Pre-holdout gates** (must pass before the holdout run is authorized):
1. **FOLDS_A CV mean R²** beats the freshly-measured lag-only GRU
   baseline (same protocol, FOLDS_A) by at least
   `0.5 · std(FOLDS_A R²)`. This is the decision-pool test.
2. **FOLDS_A non-lag R²** exceeds `NLF_A_ref + std(NLF_A_ref)` — the
   freshly-measured non-lag reference from §4.4. The legacy `0.01`
   threshold is deleted.
3. **FOLDS_B one-shot read (§8.1) does not trigger Kill Criteria.**
   The FOLDS_B R² is recorded but no design change may follow it.
4. **No FOLDS_B leakage.** No design-time decision (architecture,
   epochs, hyperparameters, blend weights, shrinkage curve, feature
   set, normalization stats, keep/drop of any phase) was derived
   from FOLDS_B val labels. Audit script verifies this, including
   the `folds_b_read_log` ledger showing exactly one read at §8.1.
5. **All selection decisions are fold-local.** No global feature sets,
   no cross-fold stability filters, no "stable across ≥ k of N folds"
   aggregations.
6. Leakage audit script exits clean on every pipeline module.
7. Final configuration, `holdout_gate.yaml`, `NLF_A_ref` and its std,
   FOLDS_B R², and data manifests are checked in so the run is
   reproducible from the repo alone.

**Holdout gate** (one-shot, exit-on-fail):
7. **All pass conditions in the committed `holdout_gate.yaml` hold.**
   This file is the **single source of truth** for holdout pass/fail;
   the rules themselves are defined only in §8.3 and are generated
   mechanically by `src/build_holdout_gate.py` from FOLDS_B scores,
   `NLF_A_ref`, and the predeclared constants. This document does
   not restate them here — restating would create ambiguity if the
   two descriptions ever drifted, which is exactly the failure mode
   Codex flagged in an earlier draft.
8. `holdout_gate.yaml` was committed to git **before** HOLDOUT was
   unsealed via `unlock_sealed.py`. The audit script refuses to
   authorize Phase 8 otherwise.

Anything short of all eight is no-ship. A holdout failure does not
trigger redesign in this cycle — it triggers cycle exit and a writeup.
RESERVE (1572–1698) lives in sealed storage per §8.5 and cannot be
unlocked inside this cycle under any circumstances.
