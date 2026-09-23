"""Regression metrics used for holdout and cross-validation evaluation."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE, MAPE (guarded) and R^2 for a regression forecast.

    MAPE is only computed when **every** actual value is strictly positive;
    otherwise it is reported as ``null`` because it is mathematically
    undefined (division by zero) or meaningless (sign flips).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty inputs.")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    mape = None
    if np.all(y_true > 0):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

    r2 = float(r2_score(y_true, y_pred)) if y_true.size > 1 else None

    return {"mae": mae, "rmse": rmse, "mape_percent": mape, "r2": r2}
