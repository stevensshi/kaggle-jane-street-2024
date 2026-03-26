"""Phase 2.2: Distributions & Summary Statistics.

Scans all 10 partitions with Polars (lazy) to compute:
1. Responder distributions (all 9) — quantiles, mean, std, skew, kurtosis
2. Feature summary stats — mean, std, min, max, quantiles, null %
3. Weight distribution
4. Missing value patterns per feature
"""

import polars as pl
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("data/raw/train.parquet")
REPORT_DIR = Path("reports/eda")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [f"feature_{i:02d}" for i in range(79)]
RESPONDER_COLS = [f"responder_{i}" for i in range(9)]


def load_all_data() -> pl.LazyFrame:
    return pl.scan_parquet(str(DATA_DIR / "**/*.parquet"))


def responder_stats(lf: pl.LazyFrame) -> dict:
    """Compute stats for all 9 responders."""
    print("Computing responder statistics...")
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    stats = {}
    for col in RESPONDER_COLS:
        exprs = [
            pl.col(col).count().alias("count"),
            pl.col(col).null_count().alias("null_count"),
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
            pl.col(col).skew().alias("skew"),
            pl.col(col).kurtosis().alias("kurtosis"),
        ]
        for q in quantiles:
            exprs.append(pl.col(col).quantile(q).alias(f"q{q}"))

        result = lf.select(exprs).collect()
        row = result.to_dicts()[0]
        stats[col] = {k: float(v) if v is not None else None for k, v in row.items()}

    return stats


def feature_stats(lf: pl.LazyFrame) -> dict:
    """Compute stats for all 79 features."""
    print("Computing feature statistics...")
    stats = {}
    # Process in batches to avoid memory issues
    batch_size = 20
    for i in range(0, len(FEATURE_COLS), batch_size):
        batch = FEATURE_COLS[i : i + batch_size]
        print(f"  Features {i} to {i + len(batch) - 1}...")
        for col in batch:
            exprs = [
                pl.col(col).count().alias("count"),
                pl.col(col).null_count().alias("null_count"),
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
                pl.col(col).skew().alias("skew"),
                pl.col(col).kurtosis().alias("kurtosis"),
                pl.col(col).quantile(0.01).alias("q0.01"),
                pl.col(col).quantile(0.25).alias("q0.25"),
                pl.col(col).quantile(0.50).alias("q0.50"),
                pl.col(col).quantile(0.75).alias("q0.75"),
                pl.col(col).quantile(0.99).alias("q0.99"),
            ]
            result = lf.select(exprs).collect()
            row = result.to_dicts()[0]
            stats[col] = {k: float(v) if v is not None else None for k, v in row.items()}

    return stats


def weight_stats(lf: pl.LazyFrame) -> dict:
    """Compute weight distribution stats."""
    print("Computing weight statistics...")
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    exprs = [
        pl.col("weight").count().alias("count"),
        pl.col("weight").null_count().alias("null_count"),
        pl.col("weight").mean().alias("mean"),
        pl.col("weight").std().alias("std"),
        pl.col("weight").min().alias("min"),
        pl.col("weight").max().alias("max"),
        pl.col("weight").skew().alias("skew"),
        (pl.col("weight") == 0).sum().alias("zero_count"),
    ]
    for q in quantiles:
        exprs.append(pl.col("weight").quantile(q).alias(f"q{q}"))

    result = lf.select(exprs).collect()
    row = result.to_dicts()[0]
    return {k: float(v) if v is not None else None for k, v in row.items()}


def missing_values(lf: pl.LazyFrame) -> dict:
    """Compute null % for all columns."""
    print("Computing missing value patterns...")
    all_cols = FEATURE_COLS + RESPONDER_COLS + ["weight"]
    exprs = [pl.col(c).null_count().alias(c) for c in all_cols]
    exprs.append(pl.len().alias("total_rows"))
    result = lf.select(exprs).collect()
    row = result.to_dicts()[0]
    total = row["total_rows"]
    return {
        c: {
            "null_count": int(row[c]),
            "null_pct": round(100.0 * row[c] / total, 4),
        }
        for c in all_cols
    }


def main():
    lf = load_all_data()

    report = {}
    report["responders"] = responder_stats(lf)
    report["features"] = feature_stats(lf)
    report["weight"] = weight_stats(lf)
    report["missing_values"] = missing_values(lf)

    out_path = REPORT_DIR / "distributions.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("RESPONDER SUMMARY")
    print("=" * 60)
    for name, s in report["responders"].items():
        print(f"\n{name}:")
        print(f"  mean={s['mean']:.6f}  std={s['std']:.6f}  skew={s['skew']:.3f}  kurt={s['kurtosis']:.3f}")
        print(f"  [1%={s['q0.01']:.6f}, 50%={s['q0.5']:.6f}, 99%={s['q0.99']:.6f}]")
        null_pct = 100.0 * s["null_count"] / s["count"] if s["count"] else 0
        print(f"  nulls: {s['null_count']:.0f} ({null_pct:.2f}%)")

    print("\n" + "=" * 60)
    print("WEIGHT SUMMARY")
    print("=" * 60)
    w = report["weight"]
    print(f"  mean={w['mean']:.6f}  std={w['std']:.6f}  min={w['min']:.6f}  max={w['max']:.6f}")
    print(f"  zeros: {w['zero_count']:.0f}  skew={w['skew']:.3f}")
    print(f"  [1%={w['q0.01']:.6f}, 50%={w['q0.5']:.6f}, 99%={w['q0.99']:.6f}]")

    print("\n" + "=" * 60)
    print("TOP 10 FEATURES BY NULL %")
    print("=" * 60)
    mv = report["missing_values"]
    sorted_nulls = sorted(
        [(k, v) for k, v in mv.items() if k.startswith("feature_")],
        key=lambda x: x[1]["null_pct"],
        reverse=True,
    )
    for name, v in sorted_nulls[:10]:
        print(f"  {name}: {v['null_pct']:.2f}% ({v['null_count']} nulls)")

    print("\n" + "=" * 60)
    print("TOP 10 FEATURES BY ABSOLUTE SKEW")
    print("=" * 60)
    sorted_skew = sorted(
        report["features"].items(),
        key=lambda x: abs(x[1]["skew"]) if x[1]["skew"] is not None else 0,
        reverse=True,
    )
    for name, s in sorted_skew[:10]:
        print(f"  {name}: skew={s['skew']:.3f}  kurt={s['kurtosis']:.3f}")

    print("\n" + "=" * 60)
    print("FEATURES WITH STD < 0.01 (potentially degenerate)")
    print("=" * 60)
    for name, s in sorted(report["features"].items()):
        if s["std"] is not None and s["std"] < 0.01:
            print(f"  {name}: std={s['std']:.6f}  mean={s['mean']:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
