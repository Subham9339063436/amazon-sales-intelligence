"""Supervised forecasting models: Linear Regression baseline and LightGBM."""

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LinearRegression

from .config import (
    LIGHTGBM_EARLY_STOPPING_ROUNDS,
    LIGHTGBM_PARAMS,
    LIGHTGBM_ROUNDS_CV,
    LIGHTGBM_ROUNDS_FINAL,
)


def fit_predict_linear(X_train, y_train, X_test) -> np.ndarray:
    """Fit ordinary least squares on training rows and predict test rows."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_lightgbm(
    X_train,
    y_train,
    X_test,
    X_early_stop=None,
    y_early_stop=None,
    num_boost_round: int | None = None,
    early_stopping_rounds: int | None = None,
) -> tuple[np.ndarray, lgb.Booster]:
    """Fit LightGBM and predict test rows.

    Early stopping, when requested, only ever uses ``X_early_stop`` — a tail
    carved out of the *training* period. The test set must never be passed
    here for model selection.
    """
    params = dict(LIGHTGBM_PARAMS)
    train_set = lgb.Dataset(X_train, label=y_train)

    valid_sets = None
    callbacks = []
    if X_early_stop is not None and early_stopping_rounds:
        valid_sets = [lgb.Dataset(X_early_stop, label=y_early_stop)]
        callbacks.append(
            lgb.early_stopping(early_stopping_rounds, verbose=False)
        )

    rounds = num_boost_round or LIGHTGBM_ROUNDS_FINAL
    model = lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )

    best_iteration = getattr(model, "best_iteration", 0)
    pred_kwargs = {}
    if best_iteration:
        pred_kwargs["num_iteration"] = best_iteration
    preds = model.predict(X_test, **pred_kwargs)
    return preds, model
