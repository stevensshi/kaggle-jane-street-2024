"""Metric library for Jane Street 2024.

All metrics match the competition's evaluation formula exactly.
Kaggle formula (zero-mean weighted R²):
    R² = 1 - Σ(w · (y - ŷ)²) / Σ(w · y²)

Unit tests run via:  python -m src.metrics
"""

import numpy as np


def weighted_r2(y, yhat, w) -> float:
    """Zero-mean weighted R², matching Kaggle's evaluation formula.

    The baseline is zero (not sample mean), so always predicting 0 gives R²=0.
    A model worse than predicting zero gives negative R².
    """
    y    = np.asarray(y,    dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    w    = np.asarray(w,    dtype=np.float64)
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * y ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def cv_score(fold_r2s: list[float], lam: float = 0.5) -> float:
    """Penalised CV score: mean(R²) − lam · std(R²).

    Rewards consistent performance across folds, not just average.
    """
    arr = np.array(fold_r2s, dtype=np.float64)
    return float(arr.mean() - lam * arr.std())


def non_lag_r2(model, X: np.ndarray, y, w, lag_col_indices: list[int],
               mode: str = "zero") -> float:
    """R² with lag features ablated.

    Args:
        model:            fitted model with .predict(X) method.
        X:                feature matrix (n_samples, n_features).
        y, w:             target and weights.
        lag_col_indices:  column indices in X that are lag features.
        mode:             "zero"  — set lag cols to 0.
                          "noise" — replace with N(0, col_std) noise.
    Returns:
        weighted_r2 score on ablated features.
    """
    X_abl = X.copy()
    if mode == "zero":
        X_abl[:, lag_col_indices] = 0.0
    elif mode == "noise":
        rng = np.random.default_rng(seed=0)
        for idx in lag_col_indices:
            std = float(X[:, idx].std()) or 1.0
            X_abl[:, idx] = rng.normal(0.0, std, size=len(X))
    else:
        raise ValueError(f"mode must be 'zero' or 'noise', got {mode!r}")
    yhat = model.predict(X_abl)
    return weighted_r2(y, yhat, w)


# ── Unit tests ───────────────────────────────────────────────────────────────

def _run_tests():
    import sys

    failures = []

    def check(name, got, expected, tol=1e-9):
        if abs(got - expected) > tol:
            failures.append(f"FAIL {name}: got {got}, expected {expected}")
        else:
            print(f"  ok  {name}")

    # Perfect prediction → R² = 1
    y = np.array([1.0, -1.0, 2.0])
    w = np.array([1.0,  1.0, 1.0])
    check("perfect prediction", weighted_r2(y, y, w), 1.0)

    # Predict zero → R² = 0 (zero-mean baseline is zero)
    check("predict zero", weighted_r2(y, np.zeros(3), w), 0.0)

    # Predict negative of truth → R² = -3  (ss_res = 4*ss_tot)
    # ss_res = sum(w*(y-(-y))^2) = sum(w*(2y)^2) = 4*sum(w*y^2) = 4*ss_tot
    check("predict negation", weighted_r2(y, -y, w), -3.0)

    # Weighted: zero-weight rows don't count
    y2 = np.array([1.0, 999.0])
    w2 = np.array([1.0, 0.0])
    check("zero weight row ignored", weighted_r2(y2, np.zeros(2), w2), 0.0)

    # All-zero target → R² = 0 (guard against division by zero)
    check("zero target", weighted_r2(np.zeros(3), np.ones(3), w), 0.0)

    # cv_score: stable signal → penalised less
    check("cv_score mean", cv_score([0.1, 0.1, 0.1]), 0.1)
    check("cv_score penalised", cv_score([0.0, 0.2]), 0.1 - 0.5 * 0.1, tol=1e-9)

    if failures:
        for f in failures:
            print(f)
        sys.exit(1)
    else:
        print("all metrics tests passed")


if __name__ == "__main__":
    _run_tests()
