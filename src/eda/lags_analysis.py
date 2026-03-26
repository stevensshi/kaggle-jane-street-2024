"""Phase 2.5: Lags File Analysis.

1. Verify lags structure
2. Correlation of lagged responders with current responder_6
3. How predictive is responder_6_lag_1 alone? (naive baseline R²)
"""

import polars as pl
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import weighted_r2

DATA_DIR = Path("data/raw/train.parquet")
LAGS_FILE = Path("data/raw/lags.parquet")
REPORT_DIR = Path("reports/eda")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_lags_structure() -> dict:
    """Verify lags file structure."""
    print("Analyzing lags file structure...")
    lags = pl.read_parquet(str(LAGS_FILE))
    print(f"  Shape: {lags.shape}")
    print(f"  Columns: {lags.columns}")
    print(f"  Dtypes: {lags.dtypes}")

    return {
        "shape": list(lags.shape),
        "columns": lags.columns,
        "dtypes": [str(d) for d in lags.dtypes],
        "head": lags.head(5).to_dicts(),
    }


def lag_target_correlations() -> dict:
    """Correlate lagged responders with current responder_6.

    The lags file tells us which responder columns are available as lags.
    During inference, we get lagged responders from the *previous* time slot.
    We need to join train data with shifted responders to simulate this.
    """
    print("Computing lag-target correlations...")

    lf = pl.scan_parquet(str(DATA_DIR / "**/*.parquet"))

    # Within each (date_id, symbol_id), shift responders by 1 time step
    # to create lagged versions, then correlate with current responder_6
    # Sample dates for efficiency
    all_dates = lf.select("date_id").unique().sort("date_id").collect()
    dates = all_dates["date_id"].to_numpy()
    sample_idx = np.linspace(0, len(dates) - 1, 100, dtype=int)
    sample_dates = dates[sample_idx].tolist()

    responder_cols = [f"responder_{i}" for i in range(9)]
    corrs = {f"lag1_{c}": [] for c in responder_cols}

    select_cols = list(dict.fromkeys(["time_id", "symbol_id", "responder_6"] + responder_cols))

    for d in sample_dates:
        day_data = (
            lf.filter(pl.col("date_id") == d)
            .sort("time_id", "symbol_id")
            .select(select_cols)
            .collect()
        )

        if len(day_data) == 0:
            continue

        # Create lag-1 within each symbol
        for resp in responder_cols:
            lagged = day_data.with_columns(
                pl.col(resp).shift(1).over("symbol_id").alias(f"lag1_{resp}")
            ).drop_nulls(f"lag1_{resp}")

            if len(lagged) < 10:
                continue

            x = lagged[f"lag1_{resp}"].to_numpy()
            y = lagged["responder_6"].to_numpy()
            if np.std(x) > 0 and np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
                corrs[f"lag1_{resp}"].append(float(corr))

    result = {}
    for key, vals in corrs.items():
        if vals:
            result[key] = {
                "mean_corr": round(float(np.mean(vals)), 6),
                "std_corr": round(float(np.std(vals)), 6),
                "n_days": len(vals),
            }

    return result


def naive_lag_baseline() -> dict:
    """Compute R² using responder_6_lag_1 as the prediction.

    Simulates what we'd get in inference by shifting responder_6 by 1 time step.
    Uses validation dates (1189-1443) as proposed in temporal analysis.
    """
    print("Computing naive lag baseline R²...")

    lf = pl.scan_parquet(str(DATA_DIR / "**/*.parquet"))

    # Use validation period
    val_data = (
        lf.filter((pl.col("date_id") >= 1189) & (pl.col("date_id") <= 1443))
        .sort("date_id", "time_id", "symbol_id")
        .select(["date_id", "time_id", "symbol_id", "responder_6", "weight"])
        .collect()
    )

    # Create lag-1 of responder_6 within each symbol
    val_with_lag = val_data.with_columns(
        pl.col("responder_6").shift(1).over("symbol_id").alias("lag1_resp6")
    ).drop_nulls("lag1_resp6")

    y_true = val_with_lag["responder_6"].to_numpy()
    y_pred = val_with_lag["lag1_resp6"].to_numpy()
    weight = val_with_lag["weight"].to_numpy()

    r2_lag1 = weighted_r2(y_true, y_pred, weight)
    r2_zero = weighted_r2(y_true, np.zeros_like(y_true), weight)

    # Also try scaled lag (multiply by optimal coefficient)
    # Optimal scaling: alpha = sum(w * y * lag) / sum(w * lag^2)
    alpha = np.sum(weight * y_true * y_pred) / np.sum(weight * y_pred ** 2)
    r2_scaled = weighted_r2(y_true, alpha * y_pred, weight)

    return {
        "validation_dates": "1189-1443",
        "n_rows": len(val_with_lag),
        "r2_predict_zero": round(float(r2_zero), 8),
        "r2_predict_lag1": round(float(r2_lag1), 8),
        "r2_predict_scaled_lag1": round(float(r2_scaled), 8),
        "optimal_scale": round(float(alpha), 6),
    }


def main():
    report = {}
    report["lags_structure"] = analyze_lags_structure()
    report["lag_target_correlations"] = lag_target_correlations()
    report["naive_baseline"] = naive_lag_baseline()

    out_path = REPORT_DIR / "lags_analysis.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("LAGS FILE STRUCTURE")
    print("=" * 60)
    ls = report["lags_structure"]
    print(f"  Shape: {ls['shape']}")
    print(f"  Columns: {ls['columns']}")

    print("\n" + "=" * 60)
    print("LAG-1 RESPONDER vs CURRENT RESPONDER_6 CORRELATIONS")
    print("=" * 60)
    ltc = report["lag_target_correlations"]
    sorted_ltc = sorted(ltc.items(), key=lambda x: abs(x[1]["mean_corr"]), reverse=True)
    for key, v in sorted_ltc:
        print(f"  {key}: mean_corr={v['mean_corr']:+.6f}  std={v['std_corr']:.6f}")

    print("\n" + "=" * 60)
    print("NAIVE BASELINE R² (validation set)")
    print("=" * 60)
    nb = report["naive_baseline"]
    print(f"  Predict zero:       R² = {nb['r2_predict_zero']:.8f}")
    print(f"  Predict lag1:       R² = {nb['r2_predict_lag1']:.8f}")
    print(f"  Predict scaled lag: R² = {nb['r2_predict_scaled_lag1']:.8f}  (scale={nb['optimal_scale']:.4f})")
    print(f"  N rows: {nb['n_rows']:,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
