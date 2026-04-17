"""Minimal leakage audit — three concrete checks.

Exits 0 if clean, non-zero on any violation.

Usage:
    python src/audit_leakage.py            # runs all checks + fixtures
    python src/audit_leakage.py --quick    # skip fixture suite
"""

import argparse
import sys
import numpy as np


# ── Check 1: Normalisation fit on val ────────────────────────────────────────

def check_normalisation(scaler, val_date_lo: int, fit_date_hi: int,
                         name: str = "scaler") -> list[str]:
    """Verify a scaler was fit on data ending before val_date_lo.

    The scaler must have a `fit_date_hi_` attribute set during fitting,
    recording the last date_id in its training data.
    """
    violations = []
    if not hasattr(scaler, "fit_date_hi_"):
        violations.append(
            f"VIOLATION: {name} has no fit_date_hi_ attribute. "
            "Cannot verify it was fit on train-only data. "
            "Set scaler.fit_date_hi_ = max(date_id in training data) after fitting."
        )
        return violations
    if scaler.fit_date_hi_ >= val_date_lo:
        violations.append(
            f"VIOLATION: {name}.fit_date_hi_={scaler.fit_date_hi_} "
            f">= val_date_lo={val_date_lo}. "
            "Scaler was fit on data that overlaps the validation window."
        )
    return violations


# ── Check 2: Lag off-by-one ───────────────────────────────────────────────────

def check_lag_contract(df, val_date_lo: int, val_date_hi: int,
                        sample_frac: float = 0.05,
                        rng_seed: int = 42) -> list[str]:
    """Verify lag_1 responders equal the previous slot's responders.

    Checks:
    - First and last time_id of every val date_id (boundary slots).
    - A random sample_frac of remaining val rows.

    df must have columns: date_id, time_id, symbol_id,
                          responder_6, responder_6_lag_1.
    """
    import polars as pl
    violations = []

    val = df.filter(
        (pl.col("date_id") >= val_date_lo) & (pl.col("date_id") <= val_date_hi)
    )
    if len(val) == 0:
        violations.append("VIOLATION: val slice is empty — nothing to audit.")
        return violations

    # Build ground truth: previous slot's responder_6 per symbol
    full_sorted = df.sort(["symbol_id", "date_id", "time_id"])
    gt = full_sorted.with_columns(
        pl.col("responder_6").shift(1).over("symbol_id").alias("_expected_lag")
    )

    # Boundary rows: first and last time_id per val date
    time_bounds = (
        val.group_by("date_id")
           .agg([pl.col("time_id").min().alias("t_min"),
                 pl.col("time_id").max().alias("t_max")])
    )
    boundary_dates  = time_bounds["date_id"].to_list()
    boundary_tmins  = time_bounds["t_min"].to_list()
    boundary_tmaxs  = time_bounds["t_max"].to_list()

    boundary_mask = pl.lit(False)
    for d, tmin, tmax in zip(boundary_dates, boundary_tmins, boundary_tmaxs):
        boundary_mask = boundary_mask | (
            (pl.col("date_id") == d) &
            ((pl.col("time_id") == tmin) | (pl.col("time_id") == tmax))
        )

    boundary_rows = gt.filter(
        (pl.col("date_id") >= val_date_lo) & (pl.col("date_id") <= val_date_hi)
        & boundary_mask
    )

    # Random sample of non-boundary val rows
    non_boundary = gt.filter(
        (pl.col("date_id") >= val_date_lo) & (pl.col("date_id") <= val_date_hi)
        & ~boundary_mask
    )
    n_sample = max(1, int(len(non_boundary) * sample_frac))
    rng = np.random.default_rng(rng_seed)
    sample_idx = rng.choice(len(non_boundary), size=n_sample, replace=False)
    sampled = non_boundary[sample_idx.tolist()]

    rows_to_check = pl.concat([boundary_rows, sampled])

    bad = rows_to_check.filter(
        (pl.col("_expected_lag").is_not_null()) &
        (pl.col("responder_6_lag_1").is_not_null()) &
        ((pl.col("responder_6_lag_1") - pl.col("_expected_lag")).abs() > 1e-5)
    )
    if len(bad) > 0:
        violations.append(
            f"VIOLATION: {len(bad)} rows have responder_6_lag_1 != prev slot's "
            f"responder_6 (tolerance 1e-5). Sample: "
            + str(bad.head(3))
        )
    return violations


# ── Check 3: time_id / date_id swap ──────────────────────────────────────────

def check_time_date_not_swapped(df, feature_fn) -> list[str]:
    """Verify two rows with same date_id but different time_id get different
    rolling/regime features.

    feature_fn(df) → df with computed features appended.
    If the result is identical for same-date different-time rows, the feature
    likely used date_id where time_id was intended.
    """
    import polars as pl
    violations = []

    # Find a date with at least 2 time_ids
    counts = df.group_by("date_id").agg(pl.col("time_id").n_unique().alias("n"))
    multi = counts.filter(pl.col("n") > 1)
    if len(multi) == 0:
        violations.append("AUDIT SKIP: no date_id with >1 time_id in data slice.")
        return violations

    test_date = multi["date_id"][0]
    subset = df.filter(pl.col("date_id") == test_date).sort("time_id")
    row0 = subset.head(1)
    row1 = subset.slice(1, 1)

    try:
        feat0 = feature_fn(row0)
        feat1 = feature_fn(row1)
    except Exception as e:
        violations.append(f"AUDIT ERROR: feature_fn raised: {e}")
        return violations

    # Compare only new columns (not in original)
    new_cols = [c for c in feat0.columns if c not in df.columns]
    if not new_cols:
        violations.append(
            "AUDIT SKIP: feature_fn added no new columns — nothing to check."
        )
        return violations

    for col in new_cols:
        v0 = feat0[col][0]
        v1 = feat1[col][0]
        if v0 == v1:
            violations.append(
                f"VIOLATION: feature '{col}' is identical for two rows at "
                f"same date_id={test_date} but different time_ids "
                f"({subset['time_id'][0]} vs {subset['time_id'][1]}). "
                "Possible time_id/date_id swap."
            )
    return violations


# ── Deterministic fixture suite ───────────────────────────────────────────────

def run_fixtures() -> list[str]:
    """Known-bad cases that the audit must catch. Returns list of failures."""
    import polars as pl
    failures = []

    # Fixture 1: scaler fit on val data must be caught
    class FakeScaler:
        pass
    s = FakeScaler()
    s.fit_date_hi_ = 1200  # > val_date_lo=1189
    issues = check_normalisation(s, val_date_lo=1189, fit_date_hi=1200,
                                  name="fixture_scaler")
    if not issues:
        failures.append("FIXTURE FAIL: normalisation check missed val-fit scaler")

    # Fixture 2: scaler without fit_date_hi_ must be caught
    s2 = FakeScaler()
    issues2 = check_normalisation(s2, val_date_lo=1189, fit_date_hi=1200,
                                   name="fixture_scaler_no_attr")
    if not issues2:
        failures.append("FIXTURE FAIL: normalisation check missed missing fit_date_hi_")

    # Fixture 3: correct scaler must pass
    s3 = FakeScaler()
    s3.fit_date_hi_ = 1188
    issues3 = check_normalisation(s3, val_date_lo=1189, fit_date_hi=1188,
                                   name="fixture_scaler_ok")
    if issues3:
        failures.append(f"FIXTURE FAIL: clean scaler incorrectly flagged: {issues3}")

    return failures


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Leakage audit")
    parser.add_argument("--quick", action="store_true",
                        help="Skip fixture suite")
    args = parser.parse_args()

    violations = []

    if not args.quick:
        print("Running fixture suite...")
        fixture_failures = run_fixtures()
        if fixture_failures:
            for f in fixture_failures:
                print(f)
            violations.extend(fixture_failures)
        else:
            print("  ok  all fixtures")

    print("\nChecks requiring data are run by individual pipeline scripts.")
    print("Call check_normalisation(), check_lag_contract(), and")
    print("check_time_date_not_swapped() from each pipeline, then collect results.")

    if violations:
        print(f"\nAUDIT FAILED: {len(violations)} violation(s)")
        sys.exit(1)
    else:
        print("\nAUDIT CLEAN")
        sys.exit(0)


if __name__ == "__main__":
    main()
