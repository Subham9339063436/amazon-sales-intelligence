"""Tests for leakage-free feature engineering (the critical preprocessing)."""

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.features import add_lag_features
from tests.conftest import make_panel


def test_rolling_excludes_current_row():
    """sales_rolling_3 must equal mean(t-3..t-1), never including t."""
    panel = make_panel({"A": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]})
    out = add_lag_features(panel)
    s = out["sales_rolling_3"]

    # First three rows have fewer than three prior observations -> NaN.
    assert s.iloc[:3].isna().all()
    # Row 3 window = mean(rows 0,1,2) = mean(3,6,9) = 6.0
    assert s.iloc[3] == 6.0
    # Row 4 window = mean(6,9,12) = 9.0  (mean including current would be 11.0)
    assert s.iloc[4] == 9.0
    # Row 5 window = mean(9,12,15) = 12.0 (incl. current would be 15.0)
    assert s.iloc[5] == 12.0


def test_rolling_value_changes_when_only_current_target_changes():
    """If only the current target changes, the rolling feature must not move.

    This is the direct leakage regression test: in the original (leaky)
    implementation, changing row t's target changed row t's rolling feature.
    """
    base = make_panel({"A": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]})
    perturbed_raw = [3.0, 6.0, 9.0, 12.0, 99999.0, 18.0]
    perturbed = make_panel({"A": perturbed_raw})

    base_out = add_lag_features(base)
    pert_out = add_lag_features(perturbed)

    # Row 4's own target changed...
    assert pert_out.loc[4, TARGET_COLUMN] == 99999.0
    # ...but row 4's rolling feature (window = rows 1,2,3) must be identical.
    assert pert_out.loc[4, "sales_rolling_3"] == base_out.loc[4, "sales_rolling_3"]
    # Lags of row 4 also depend only on rows 1..3 and must be unchanged.
    for col in ["sales_lag_1", "sales_lag_2", "sales_lag_3"]:
        assert pert_out.loc[4, col] == base_out.loc[4, col]


def test_features_never_use_future_rows():
    """Changing a future target must not change earlier rows' features."""
    base = make_panel({"A": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]})
    perturbed = make_panel({"A": [3.0, 6.0, 9.0, 12.0, 15.0, 99999.0]})

    base_out = add_lag_features(base)
    pert_out = add_lag_features(perturbed)

    feature_cols = ["sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_rolling_3"]
    # All rows up to and including the changed row must be identical
    # (row 5's features only look at rows 2..4).
    pd.testing.assert_frame_equal(
        base_out.loc[:4, feature_cols],
        pert_out.loc[:4, feature_cols],
    )


def test_lags_are_correct_and_per_product():
    """Lags shift within product boundaries, not across products."""
    panel = make_panel({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        "B": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    out = add_lag_features(panel)
    a = out[out["product"] == "A"].reset_index(drop=True)
    b = out[out["product"] == "B"].reset_index(drop=True)

    # Row 3 (0-based) of A has value 4.0: lag1=row2, lag2=row1, lag3=row0.
    assert a["sales_lag_1"].iloc[2] == 2.0
    assert a["sales_lag_2"].iloc[3] == 2.0
    assert a["sales_lag_3"].iloc[5] == 3.0
    # Product B's first row must be NaN, not product A's last value.
    assert np.isnan(b["sales_lag_1"].iloc[0])


def test_modeling_frame_has_no_nan_features(tiny_modeling):
    """The frame fed to models (after dropna) must contain no NaN features."""
    clean = tiny_modeling.dropna(subset=FEATURE_COLUMNS)
    assert len(clean) > 0
    assert not clean[FEATURE_COLUMNS].isna().any().any()
    assert not clean[TARGET_COLUMN].isna().any()


def test_deterministic_given_input():
    panel = make_panel({"A": list(range(1, 11))})
    pd.testing.assert_frame_equal(add_lag_features(panel), add_lag_features(panel))
