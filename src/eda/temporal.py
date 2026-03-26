"""Phase 2.3: Temporal Structure Analysis.

1. Time slots per day across all dates
2. Symbols per time slot
3. Stationarity check: feature distributions by date range
4. Responder_6 mean/std per date range — regime changes
5. Auto-correlation of responder_6
6. Propose train / validation / holdout date boundaries
"""

import polars as pl
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("data/raw/train.parquet")
REPORT_DIR = Path("reports/eda")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_data() -> pl.LazyFrame:
    return pl.scan_parquet(str(DATA_DIR / "**/*.parquet"))


def time_structure(lf: pl.LazyFrame) -> dict:
    """Analyze time slots per day and symbols per time slot."""
    print("Analyzing time structure...")

    # Time slots per day
    slots_per_day = (
        lf.group_by("date_id")
        .agg(pl.col("time_id").n_unique().alias("n_time_slots"))
        .sort("date_id")
        .collect()
    )

    slot_counts = slots_per_day["n_time_slots"].to_numpy()
    result = {
        "time_slots_per_day": {
            "mean": float(np.mean(slot_counts)),
            "std": float(np.std(slot_counts)),
            "min": int(np.min(slot_counts)),
            "max": int(np.max(slot_counts)),
            "median": float(np.median(slot_counts)),
            "n_days": len(slot_counts),
        }
    }

    # Symbols per time slot (sample a few days)
    symbols_per_slot = (
        lf.group_by("date_id", "time_id")
        .agg(pl.col("symbol_id").n_unique().alias("n_symbols"))
        .collect()
    )
    sym_counts = symbols_per_slot["n_symbols"].to_numpy()
    result["symbols_per_time_slot"] = {
        "mean": float(np.mean(sym_counts)),
        "std": float(np.std(sym_counts)),
        "min": int(np.min(sym_counts)),
        "max": int(np.max(sym_counts)),
        "median": float(np.median(sym_counts)),
    }

    # Date range
    date_range = lf.select(
        pl.col("date_id").min().alias("min_date"),
        pl.col("date_id").max().alias("max_date"),
    ).collect()
    result["date_range"] = {
        "min": int(date_range["min_date"][0]),
        "max": int(date_range["max_date"][0]),
    }

    return result


def responder6_by_date_range(lf: pl.LazyFrame, n_buckets: int = 10) -> dict:
    """Responder_6 mean/std per date bucket — detect regime changes."""
    print("Analyzing responder_6 by date range...")

    # Get date range
    date_range = lf.select(
        pl.col("date_id").min().alias("min"),
        pl.col("date_id").max().alias("max"),
    ).collect()
    min_d, max_d = int(date_range["min"][0]), int(date_range["max"][0])

    bucket_size = (max_d - min_d + 1) / n_buckets
    results = []

    for i in range(n_buckets):
        start = int(min_d + i * bucket_size)
        end = int(min_d + (i + 1) * bucket_size) - 1
        if i == n_buckets - 1:
            end = max_d

        stats = (
            lf.filter((pl.col("date_id") >= start) & (pl.col("date_id") <= end))
            .select(
                pl.col("responder_6").mean().alias("mean"),
                pl.col("responder_6").std().alias("std"),
                pl.col("responder_6").skew().alias("skew"),
                pl.col("responder_6").kurtosis().alias("kurtosis"),
                pl.col("weight").mean().alias("weight_mean"),
                pl.len().alias("count"),
            )
            .collect()
        )
        row = stats.to_dicts()[0]
        results.append({
            "date_start": start,
            "date_end": end,
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "skew": float(row["skew"]),
            "kurtosis": float(row["kurtosis"]),
            "weight_mean": float(row["weight_mean"]),
        })

    return {"buckets": results, "n_buckets": n_buckets}


def feature_stationarity(lf: pl.LazyFrame) -> dict:
    """Check if feature distributions shift across date ranges.
    Compare mean/std of each feature in first third vs last third of data.
    """
    print("Checking feature stationarity...")

    date_range = lf.select(
        pl.col("date_id").min().alias("min"),
        pl.col("date_id").max().alias("max"),
    ).collect()
    min_d, max_d = int(date_range["min"][0]), int(date_range["max"][0])
    third = (max_d - min_d) // 3

    early_end = min_d + third
    late_start = max_d - third

    feature_cols = [f"feature_{i:02d}" for i in range(79)]

    # Compute mean/std for early and late periods
    early_stats = (
        lf.filter(pl.col("date_id") <= early_end)
        .select(
            [pl.col(c).mean().alias(f"{c}_mean") for c in feature_cols]
            + [pl.col(c).std().alias(f"{c}_std") for c in feature_cols]
        )
        .collect()
    )

    late_stats = (
        lf.filter(pl.col("date_id") >= late_start)
        .select(
            [pl.col(c).mean().alias(f"{c}_mean") for c in feature_cols]
            + [pl.col(c).std().alias(f"{c}_std") for c in feature_cols]
        )
        .collect()
    )

    shifts = {}
    for c in feature_cols:
        e_mean = float(early_stats[f"{c}_mean"][0]) if early_stats[f"{c}_mean"][0] is not None else 0
        l_mean = float(late_stats[f"{c}_mean"][0]) if late_stats[f"{c}_mean"][0] is not None else 0
        e_std = float(early_stats[f"{c}_std"][0]) if early_stats[f"{c}_std"][0] is not None else 1
        l_std = float(late_stats[f"{c}_std"][0]) if late_stats[f"{c}_std"][0] is not None else 1

        # Measure shift as absolute difference in means / pooled std
        pooled_std = (e_std + l_std) / 2.0 if (e_std + l_std) > 0 else 1.0
        mean_shift = abs(l_mean - e_mean) / pooled_std
        std_ratio = l_std / e_std if e_std > 0 else float("inf")

        shifts[c] = {
            "early_mean": e_mean,
            "late_mean": l_mean,
            "early_std": e_std,
            "late_std": l_std,
            "mean_shift_z": round(mean_shift, 4),
            "std_ratio": round(std_ratio, 4),
        }

    return {
        "early_dates": f"{min_d}-{early_end}",
        "late_dates": f"{late_start}-{max_d}",
        "shifts": shifts,
    }


def responder6_autocorrelation(lf: pl.LazyFrame, max_lag: int = 5) -> dict:
    """Autocorrelation of responder_6 within days.
    Sample a subset of days for efficiency.
    """
    print("Computing responder_6 autocorrelation...")

    # Sample 50 days spread across the dataset
    all_dates = lf.select("date_id").unique().sort("date_id").collect()
    dates = all_dates["date_id"].to_numpy()
    sample_idx = np.linspace(0, len(dates) - 1, 50, dtype=int)
    sample_dates = dates[sample_idx].tolist()

    autocorrs = {lag: [] for lag in range(1, max_lag + 1)}

    for d in sample_dates:
        day_data = (
            lf.filter(pl.col("date_id") == d)
            .sort("time_id", "symbol_id")
            .select("responder_6")
            .collect()
        )
        vals = day_data["responder_6"].to_numpy()
        if len(vals) < max_lag + 10:
            continue

        for lag in range(1, max_lag + 1):
            x = vals[:-lag]
            y = vals[lag:]
            if np.std(x) > 0 and np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
                autocorrs[lag].append(float(corr))

    result = {}
    for lag, corrs in autocorrs.items():
        if corrs:
            result[f"lag_{lag}"] = {
                "mean": float(np.mean(corrs)),
                "std": float(np.std(corrs)),
                "min": float(np.min(corrs)),
                "max": float(np.max(corrs)),
            }

    return result


def propose_splits(lf: pl.LazyFrame) -> dict:
    """Propose train/validation/holdout date boundaries."""
    date_range = lf.select(
        pl.col("date_id").min().alias("min"),
        pl.col("date_id").max().alias("max"),
    ).collect()
    min_d, max_d = int(date_range["min"][0]), int(date_range["max"][0])
    total = max_d - min_d + 1

    # 70% train, 15% validation, 15% holdout
    train_end = min_d + int(total * 0.70) - 1
    val_end = min_d + int(total * 0.85) - 1
    holdout_start = val_end + 1

    return {
        "train": {"start": min_d, "end": train_end, "n_days": train_end - min_d + 1},
        "validation": {"start": train_end + 1, "end": val_end, "n_days": val_end - train_end},
        "holdout": {"start": holdout_start, "end": max_d, "n_days": max_d - holdout_start + 1},
        "total_days": total,
    }


def main():
    lf = load_all_data()

    report = {}
    report["time_structure"] = time_structure(lf)
    report["responder6_by_date"] = responder6_by_date_range(lf)
    report["stationarity"] = feature_stationarity(lf)
    report["autocorrelation"] = responder6_autocorrelation(lf)
    report["proposed_splits"] = propose_splits(lf)

    out_path = REPORT_DIR / "temporal.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")

    # Print summary
    ts = report["time_structure"]
    print("\n" + "=" * 60)
    print("TIME STRUCTURE")
    print("=" * 60)
    print(f"  Date range: {ts['date_range']['min']} to {ts['date_range']['max']}")
    print(f"  Days: {ts['time_slots_per_day']['n_days']}")
    tspd = ts["time_slots_per_day"]
    print(f"  Time slots/day: mean={tspd['mean']:.1f} std={tspd['std']:.1f} min={tspd['min']} max={tspd['max']}")
    spst = ts["symbols_per_time_slot"]
    print(f"  Symbols/slot: mean={spst['mean']:.1f} std={spst['std']:.1f} min={spst['min']} max={spst['max']}")

    print("\n" + "=" * 60)
    print("RESPONDER_6 BY DATE RANGE (regime check)")
    print("=" * 60)
    for b in report["responder6_by_date"]["buckets"]:
        print(f"  [{b['date_start']:4d}-{b['date_end']:4d}] "
              f"mean={b['mean']:+.5f} std={b['std']:.4f} skew={b['skew']:+.3f} "
              f"weight_mean={b['weight_mean']:.3f} n={b['count']:,}")

    print("\n" + "=" * 60)
    print("FEATURE STATIONARITY — TOP 10 SHIFTED FEATURES")
    print("=" * 60)
    shifts = report["stationarity"]["shifts"]
    sorted_shifts = sorted(shifts.items(), key=lambda x: x[1]["mean_shift_z"], reverse=True)
    for name, s in sorted_shifts[:10]:
        print(f"  {name}: shift_z={s['mean_shift_z']:.4f}  std_ratio={s['std_ratio']:.4f}"
              f"  early_mean={s['early_mean']:.4f}  late_mean={s['late_mean']:.4f}")

    print("\n" + "=" * 60)
    print("RESPONDER_6 AUTOCORRELATION")
    print("=" * 60)
    for lag, stats in report["autocorrelation"].items():
        print(f"  {lag}: mean={stats['mean']:.4f} std={stats['std']:.4f}")

    print("\n" + "=" * 60)
    print("PROPOSED DATA SPLITS")
    print("=" * 60)
    sp = report["proposed_splits"]
    print(f"  Train:      {sp['train']['start']}-{sp['train']['end']} ({sp['train']['n_days']} days)")
    print(f"  Validation: {sp['validation']['start']}-{sp['validation']['end']} ({sp['validation']['n_days']} days)")
    print(f"  Holdout:    {sp['holdout']['start']}-{sp['holdout']['end']} ({sp['holdout']['n_days']} days)")
    print(f"  Total:      {sp['total_days']} days")

    print("\nDone!")


if __name__ == "__main__":
    main()
