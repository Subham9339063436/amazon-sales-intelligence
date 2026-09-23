"""Chronological train/test splitting and walk-forward validation by month."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import MONTH_COLUMN, TEST_FRACTION, CV_SPLITS


def _fmt_month(month) -> str:
    return pd.Timestamp(month).strftime("%Y-%m")


def chronological_split(
    frame: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    month_col: str = MONTH_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split rows so training months strictly precede test months.

    Splits on the sorted list of *observed* months: the last
    ``test_fraction`` of months become the holdout test set. Guarantees
    ``max(train month) < min(test month)``.
    """
    months = np.sort(frame[month_col].unique())
    if len(months) < 2:
        raise ValueError("Need at least two distinct months to split.")

    cut = int(len(months) * (1 - test_fraction))
    cut = max(1, min(cut, len(months) - 1))
    train_months, test_months = months[:cut], months[cut:]

    train = frame[frame[month_col].isin(train_months)].reset_index(drop=True)
    test = frame[frame[month_col].isin(test_months)].reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")
    if train[month_col].max() >= test[month_col].min():
        raise AssertionError("Train months must strictly precede test months.")

    info = {
        "method": "chronological split on observed months",
        "test_fraction": test_fraction,
        "n_months_total": int(len(months)),
        "n_train_months": int(len(train_months)),
        "n_test_months": int(len(test_months)),
        "train_month_range": [_fmt_month(train_months[0]), _fmt_month(train_months[-1])],
        "test_month_range": [_fmt_month(test_months[0]), _fmt_month(test_months[-1])],
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
    }
    return train, test, info


def walk_forward_folds(
    frame: pd.DataFrame,
    n_splits: int = CV_SPLITS,
    month_col: str = MONTH_COLUMN,
):
    """Yield expanding-window walk-forward folds defined on months.

    Uses sklearn's ``TimeSeriesSplit`` on the sorted unique months, then maps
    each fold's months back to rows. Every fold's validation months strictly
    follow its training months, so no future information enters training.
    """
    months = np.sort(frame[month_col].unique())
    if len(months) < n_splits + 1:
        raise ValueError("Not enough months for the requested number of folds.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold_id, (train_pos, val_pos) in enumerate(tscv.split(months), start=1):
        train_months = months[train_pos]
        val_months = months[val_pos]
        if train_months.max() >= val_months.min():
            raise AssertionError("Fold training months must precede validation months.")
        yield {
            "fold": fold_id,
            "train": frame[frame[month_col].isin(train_months)],
            "val": frame[frame[month_col].isin(val_months)],
            "train_range": [_fmt_month(train_months[0]), _fmt_month(train_months[-1])],
            "val_range": [_fmt_month(val_months[0]), _fmt_month(val_months[-1])],
            "n_train_months": int(len(train_months)),
            "n_val_months": int(len(val_months)),
        }
