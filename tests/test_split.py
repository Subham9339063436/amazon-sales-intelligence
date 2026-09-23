"""Tests for chronological splitting and walk-forward folds."""

import numpy as np
import pandas as pd
import pytest

from src.config import MONTH_COLUMN
from src.split import chronological_split, walk_forward_folds
from tests.conftest import make_panel


@pytest.fixture
def month_frame() -> pd.DataFrame:
    """12 monthly rows (one product), 2019-01 .. 2019-12."""
    return make_panel({"A": [float(i) for i in range(1, 13)]})


def test_train_strictly_precedes_test(month_frame):
    train, test, info = chronological_split(month_frame, test_fraction=0.25)
    assert train[MONTH_COLUMN].max() < test[MONTH_COLUMN].min()
    # 12 months * 0.75 = 9 train months, 3 test months
    assert info["n_train_months"] == 9
    assert info["n_test_months"] == 3
    assert len(train) + len(test) == len(month_frame)


def test_split_preserves_all_rows(month_frame):
    """Train + test together must contain exactly the original rows."""
    train, test, _ = chronological_split(month_frame, test_fraction=0.2)
    assert len(train) + len(test) == len(month_frame)
    combined = pd.concat([train, test]).sort_values("month").reset_index(drop=True)
    expected = month_frame.sort_values("month").reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, expected)


def test_split_respects_fraction_boundaries(month_frame):
    train, test, _ = chronological_split(month_frame, test_fraction=0.5)
    assert len(train) == 6
    assert len(test) == 6
    assert test[MONTH_COLUMN].min() == pd.Timestamp("2019-07-01")


def test_walk_forward_folds_are_chronological(month_frame):
    folds = list(walk_forward_folds(month_frame, n_splits=3))
    assert len(folds) == 3
    for fold in folds:
        tr, va = fold["train"], fold["val"]
        assert len(tr) > 0 and len(va) > 0
        assert tr[MONTH_COLUMN].max() < va[MONTH_COLUMN].min()
        # Validation months must be later than *all* training months,
        # and folds must expand: each fold trains on more months.
    train_sizes = [f["n_train_months"] for f in folds]
    assert train_sizes == sorted(train_sizes)
    assert folds[-1]["n_train_months"] > folds[0]["n_train_months"]


def test_walk_forward_validation_months_disjoint_from_training(month_frame):
    for fold in walk_forward_folds(month_frame, n_splits=2):
        tr_months = set(fold["train"][MONTH_COLUMN])
        va_months = set(fold["val"][MONTH_COLUMN])
        assert tr_months.isdisjoint(va_months)


def test_split_rejects_single_month():
    frame = make_panel({"A": [1.0]})
    with pytest.raises(ValueError):
        chronological_split(frame, test_fraction=0.2)


def test_split_works_with_dataset_gap():
    """The real data misses 2018-04..2018-12; split must still be ordered."""
    months = (
        list(pd.date_range("2017-01-01", periods=15, freq="MS"))
        + list(pd.date_range("2019-01-01", periods=12, freq="MS"))
    )
    frame = pd.DataFrame(
        {
            "product": "A",
            "month": months,
            "sales_amount": np.arange(len(months), dtype=float),
            "year": [m.year for m in months],
            "month_num": [m.month for m in months],
            "quarter": [m.quarter for m in months],
            "avg_price": 10.0,
            "list_price": 50.0,
        }
    )
    train, test, info = chronological_split(frame, test_fraction=0.2)
    assert train[MONTH_COLUMN].max() < test[MONTH_COLUMN].min()
    assert info["n_months_total"] == 27
