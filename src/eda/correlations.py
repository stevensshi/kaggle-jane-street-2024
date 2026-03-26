"""Phase 2.4: Feature-Target Relationships.

1. Correlation of all 79 features with responder_6
2. Correlation stability: early vs late dates
3. Inter-feature correlation matrix (top correlations)
4. Inter-responder correlations
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


def feature_target_correlations(lf: pl.LazyFrame) -> dict:
    """Pearson correlation of each feature with responder_6."""
    print("Computing feature-target correlations...")

    # Compute correlations in batches
    corrs = {}
    batch_size = 20
    for i in range(0, len(FEATURE_COLS), batch_size):
        batch = FEATURE_COLS[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}...")
        exprs = [pl.corr("responder_6", c).alias(c) for c in batch]
        result = lf.select(exprs).collect()
        for c in batch:
            val = result[c][0]
            corrs[c] = float(val) if val is not None else None

    return corrs


def correlation_stability(lf: pl.LazyFrame) -> dict:
    """Compare feature-target correlations in early vs late dates."""
    print("Computing correlation stability...")

    date_range = lf.select(
        pl.col("date_id").min().alias("min"),
        pl.col("date_id").max().alias("max"),
    ).collect()
    min_d, max_d = int(date_range["min"][0]), int(date_range["max"][0])
    mid = (min_d + max_d) // 2

    early_lf = lf.filter(pl.col("date_id") <= mid)
    late_lf = lf.filter(pl.col("date_id") > mid)

    early_corrs = {}
    late_corrs = {}

    batch_size = 20
    for i in range(0, len(FEATURE_COLS), batch_size):
        batch = FEATURE_COLS[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}...")

        early_exprs = [pl.corr("responder_6", c).alias(c) for c in batch]
        late_exprs = [pl.corr("responder_6", c).alias(c) for c in batch]

        early_result = early_lf.select(early_exprs).collect()
        late_result = late_lf.select(late_exprs).collect()

        for c in batch:
            ev = early_result[c][0]
            lv = late_result[c][0]
            early_corrs[c] = float(ev) if ev is not None else None
            late_corrs[c] = float(lv) if lv is not None else None

    stability = {}
    for c in FEATURE_COLS:
        e = early_corrs.get(c)
        l = late_corrs.get(c)
        if e is not None and l is not None:
            stability[c] = {
                "early": round(e, 6),
                "late": round(l, 6),
                "diff": round(abs(e - l), 6),
                "sign_consistent": (e > 0) == (l > 0) if (e != 0 and l != 0) else None,
            }

    return stability


def inter_responder_correlations(lf: pl.LazyFrame) -> dict:
    """Correlation matrix among all 9 responders."""
    print("Computing inter-responder correlations...")

    corr_matrix = {}
    for i, r1 in enumerate(RESPONDER_COLS):
        for j, r2 in enumerate(RESPONDER_COLS):
            if j <= i:
                continue
            val = lf.select(pl.corr(r1, r2)).collect().item()
            key = f"{r1}_vs_{r2}"
            corr_matrix[key] = round(float(val), 6) if val is not None else None

    return corr_matrix


def inter_feature_top_correlations(lf: pl.LazyFrame, top_n: int = 30) -> list:
    """Find the most correlated feature pairs.
    Sample data for efficiency since full 79x79 on 47M rows is heavy.
    """
    print("Computing top inter-feature correlations (sampled)...")

    # Sample ~2M rows for speed
    sampled = lf.filter(pl.col("date_id") % 10 == 0).select(FEATURE_COLS).collect()
    arr = sampled.to_numpy()

    # Compute correlation matrix with numpy (handles NaN)
    # Replace nulls with NaN for numpy
    n_features = len(FEATURE_COLS)
    corr_pairs = []

    # Compute pairwise correlations
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mask = ~(np.isnan(arr[:, i]) | np.isnan(arr[:, j]))
            if mask.sum() < 100:
                continue
            x, y = arr[mask, i], arr[mask, j]
            if np.std(x) > 0 and np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
                corr_pairs.append({
                    "f1": FEATURE_COLS[i],
                    "f2": FEATURE_COLS[j],
                    "corr": round(float(corr), 6),
                })

    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return corr_pairs[:top_n]


def main():
    lf = load_all_data()

    report = {}
    report["feature_target_corr"] = feature_target_correlations(lf)
    report["correlation_stability"] = correlation_stability(lf)
    report["inter_responder_corr"] = inter_responder_correlations(lf)
    report["top_inter_feature_corr"] = inter_feature_top_correlations(lf)

    out_path = REPORT_DIR / "correlations.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TOP 15 FEATURES BY |CORRELATION| WITH RESPONDER_6")
    print("=" * 60)
    ftc = report["feature_target_corr"]
    sorted_corr = sorted(ftc.items(), key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)
    for name, c in sorted_corr[:15]:
        stab = report["correlation_stability"].get(name, {})
        sign_ok = stab.get("sign_consistent", "?")
        print(f"  {name}: corr={c:+.6f}  early={stab.get('early', '?'):+.6f}  late={stab.get('late', '?'):+.6f}  sign_stable={sign_ok}")

    print("\n" + "=" * 60)
    print("LEAST STABLE CORRELATIONS (biggest early-late difference)")
    print("=" * 60)
    stab = report["correlation_stability"]
    sorted_stab = sorted(stab.items(), key=lambda x: x[1]["diff"], reverse=True)
    for name, s in sorted_stab[:10]:
        print(f"  {name}: early={s['early']:+.6f}  late={s['late']:+.6f}  diff={s['diff']:.6f}  sign_consistent={s['sign_consistent']}")

    print("\n" + "=" * 60)
    print("INTER-RESPONDER CORRELATIONS (with responder_6)")
    print("=" * 60)
    irc = report["inter_responder_corr"]
    r6_corrs = {k: v for k, v in irc.items() if "responder_6" in k}
    for pair, c in sorted(r6_corrs.items(), key=lambda x: abs(x[1]) if x[1] else 0, reverse=True):
        print(f"  {pair}: {c:+.6f}")

    print("\n" + "=" * 60)
    print("TOP 15 INTER-FEATURE CORRELATIONS (absolute)")
    print("=" * 60)
    for p in report["top_inter_feature_corr"][:15]:
        print(f"  {p['f1']} vs {p['f2']}: {p['corr']:+.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
