import numpy as np
import pandas as pd


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    """Competition metric: weighted R² score.

    Score of 1.0 = perfect, 0.0 = predicting zero, <0 = worse than zero.
    In practice, top solutions score ~0.005–0.01 (financial data is noisy).
    """
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    denominator = np.sum(weight * y_true**2)
    return 1 - numerator / denominator


def evaluate(df: pd.DataFrame, pred_col: str = "prediction", target_col: str = "responder_6") -> float:
    """Compute weighted R² from a dataframe with predictions."""
    return weighted_r2(
        y_true=df[target_col].values,
        y_pred=df[pred_col].values,
        weight=df["weight"].values,
    )
