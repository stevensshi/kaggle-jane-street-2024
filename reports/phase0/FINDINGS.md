# Phase 0 Findings

## Scoring Formula Verification (§0.1)

**Formula:** Zero-mean weighted R² = `1 - Σ(w·(y−ŷ)²) / Σ(w·y²)`

Old `evaluate.py` and new `src/metrics.py` produce identical results (diff < 1e-12).
The scoring formula is **not** the cause of the 60× CV-vs-LB gap.

Unit tests (7/7 passed):
- perfect prediction → 1.0
- predict zero → 0.0 (zero-mean baseline)
- predict negation → -3.0
- zero-weight rows ignored
- zero-target guard (no div/0)
- cv_score mean and penalised form

## Splits and Folds (§0.2)

| Range | Dates | Rows |
|-------|-------|------|
| TRAIN | 0–1188 | 28,339,426 |
| VAL | 1189–1443 | 9,386,696 |
| HOLDOUT | 1444–1571 | — (locked) |
| RESERVE | 1572–1698 | — (locked) |

FOLDS_A inner_split (10% tail):
- Fold 0: train=(0,450) inner_val=(451,500) eval_val=(501,700)
- Fold 1: train=(0,630) inner_val=(631,700) eval_val=(701,900)
- Fold 2: train=(0,810) inner_val=(811,900) eval_val=(901,1100)
- Fold 3: train=(0,990) inner_val=(991,1100) eval_val=(1101,1300)

FOLDS_B (one-shot at Phase 6.1): train=(0,1300) val=(1301,1443)

All ranges non-overlapping, contiguous, and FOLDS_B val < HOLDOUT.

## Data Loader (§0.3)

Provenance confirmed on first load:
- train: 28.3M rows, 92 cols, sha=3eec013e
- val: 9.4M rows, 92 cols, sha=ccda175b

FOLDS_B and HOLDOUT locks operational (raise RuntimeError without JS_PHASE_UNLOCK).

## Leakage Audit (§0.4)

Fixture suite: 3/3 pass (val-fit scaler caught, missing attr caught, clean scaler passes).

## Data Schema Notes

- 92 columns: date_id, time_id, symbol_id, weight, 79 features, 9 responders
- time_id range: 0–848 (849 intraday slots per day)
- lags.parquet is a 39-row inference template only; training lags must be
  computed by shifting responders within (symbol_id) across (date_id, time_id)
- Significant null rates in features (feature_00–04, feature_21, feature_26–27,
  feature_31 are 100% null in first 2 dates — likely late-arriving features)
- feature_09, feature_10 are Int8 (possibly categorical/encoded)

## SIGMA_MIN

SIGMA_MIN = None (to be populated by freeze_sigma_min.py after Phase 2).

## Phase 0 Gate: PASSED

- [x] Scoring formula matches Kaggle byte-for-byte
- [x] Audit script clean (fixtures all pass)
- [x] CV folds produce identical indices across reruns
- [x] splits.py committed; SIGMA_MIN = None is explicit
- [x] lb_tracker.md initialized
