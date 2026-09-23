"""Simple naive baselines used as honest reference points."""

import pandas as pd

from .config import MONTH_COLUMN, PRODUCT_COLUMN, TARGET_COLUMN


class NaiveLastValueModel:
    """Predict each product's most recently observed target value.

    A standard persistence baseline for panel forecasting: every future month
    is predicted with the last value seen for that product in training.
    Products never seen during training fall back to the global training mean
    (the fallback count is exposed via ``n_fallback_``).
    """

    def __init__(
        self,
        id_col: str = PRODUCT_COLUMN,
        month_col: str = MONTH_COLUMN,
        target: str = TARGET_COLUMN,
    ):
        self.id_col = id_col
        self.month_col = month_col
        self.target = target
        self.last_by_id_: pd.Series | None = None
        self.global_mean_: float | None = None
        self.n_fallback_: int = 0

    def fit(self, frame: pd.DataFrame) -> "NaiveLastValueModel":
        ordered = frame.sort_values(self.month_col)
        self.last_by_id_ = ordered.groupby(self.id_col)[self.target].last()
        self.global_mean_ = float(frame[self.target].mean())
        return self

    def predict(self, frame: pd.DataFrame):
        if self.last_by_id_ is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = frame[self.id_col].map(self.last_by_id_)
        self.n_fallback_ = int(preds.isna().sum())
        return preds.fillna(self.global_mean_).to_numpy(dtype=float)


def last_observed_value(train_monthly: pd.DataFrame) -> float:
    """Aggregate baseline: the last observed training-month value.

    ``train_monthly`` must be sorted-capable with a time column ``ds`` and a
    value column ``y`` (output of ``build_total_monthly_revenue``).
    """
    ordered = train_monthly.sort_values("ds")
    return float(ordered["y"].iloc[-1])
