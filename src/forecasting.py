"""Facebook Prophet forecasting on total monthly revenue."""

import pandas as pd
from prophet import Prophet

from .config import PROPHET_PARAMS
from .metrics import regression_metrics


def fit_prophet(train_monthly: pd.DataFrame) -> Prophet:
    """Fit Prophet on a ``ds``/``y`` monthly revenue frame (training only)."""
    model = Prophet(**PROPHET_PARAMS)
    model.fit(train_monthly[["ds", "y"]].copy())
    return model


def forecast_holdout(
    model: Prophet,
    train_monthly: pd.DataFrame,
    test_monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Forecast the holdout horizon and evaluate against actuals.

    Produces one forecast point per test month by generating a future frame
    covering exactly the test months, then joining on ``ds``. Returns the
    joined frame and holdout metrics.
    """
    last_train = train_monthly["ds"].max()
    n_periods = int((test_monthly["ds"] > last_train).sum())
    if n_periods <= 0:
        raise ValueError("Test months must fall after the last training month.")

    future = model.make_future_dataframe(periods=n_periods, freq="MS")
    forecast = model.predict(future)

    merged = test_monthly.merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="left",
    )
    if merged["yhat"].isna().any():
        missing = merged.loc[merged["yhat"].isna(), "ds"].tolist()
        raise RuntimeError(f"Prophet produced no forecast for: {missing}")

    metrics = regression_metrics(merged["y"].to_numpy(), merged["yhat"].to_numpy())
    return merged, metrics
