"""Canonical splits and fold definitions — single source of truth.

SIGMA_MIN is None until freeze_sigma_min.py is run at the end of Phase 2.
No other file should hard-code date ranges.
"""

# ── Date ranges ──────────────────────────────────────────────────────────────
TRAIN_RANGE   = (0, 1188)   # 1189 days, full training window
VAL_RANGE     = (1189, 1443)  # 255 days, used for model selection pre-holdout
HOLDOUT_RANGE = (1444, 1571)  # 128 days, one-shot dev holdout (honor system)
RESERVE_RANGE = (1572, 1698)  # 127 days, sealed — do NOT read this cycle

# ── Column constants ─────────────────────────────────────────────────────────
FEATURE_COLS   = [f"feature_{i:02d}" for i in range(79)]
RESPONDER_COLS = [f"responder_{i}" for i in range(9)]
LAG_COLS       = [f"responder_{i}_lag_1" for i in range(9)]
TARGET_COL     = "responder_6"
LAG_TARGET_COL = "responder_6_lag_1"

# ── Walk-forward folds ───────────────────────────────────────────────────────
# FOLDS_A: all tuning, architecture, and phase-gate decisions.
# Every pre-holdout gate in the plan reads FOLDS_A only.
FOLDS_A = [
    {"train": (0, 500),  "val": (501,  700)},
    {"train": (0, 700),  "val": (701,  900)},
    {"train": (0, 900),  "val": (901,  1100)},
    {"train": (0, 1100), "val": (1101, 1300)},
]

# FOLDS_B: outer eval, read exactly once at Phase 6.1 after full design is
# frozen on FOLDS_A. Closest distributional proxy to HOLDOUT.
FOLDS_B = [
    {"train": (0, 1300), "val": (1301, 1443)},
]


def inner_split(train_lo: int, train_hi: int, frac: float = 0.10):
    """Carve the tail ~frac of [train_lo, train_hi] as inner-val for early
    stopping. Inner-val is never the fold's val_range.

    Returns (inner_train_range, inner_val_range) as (lo, hi) tuples.
    """
    cut = train_hi - int(frac * (train_hi - train_lo))
    return (train_lo, cut), (cut + 1, train_hi)


# ── Holdout gate constant ────────────────────────────────────────────────────
# Populated by src/freeze_sigma_min.py after Phase 2 baselines.
# Frozen before any FOLDS_B read. Do not edit by hand.
SIGMA_MIN: float | None = None
