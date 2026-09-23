"""Leakage-free feature engineering for the monthly product panel.

Important correctness notes
---------------------------
* ``sales_lag_k`` uses only the product's *previous observed* months.
  The dataset contains a gap (2018-04 .. 2018-12 are missing), so "previous
  observed month" can be more than one calendar month back. Lags still only
  ever look strictly backwards in time, never forwards.
* ``sales_rolling_3`` is the mean of the **three previous observations**
  (t-3, t-2, t-1) and explicitly excludes the current row. The original
  notebook computed ``rolling(3).mean()`` *including* the current row, which
  put the target itself into the feature (target leakage).
"""

import pandas as pd

from .config import MONTH_COLUMN, PRODUCT_COLUMN, TARGET_COLUMN


def add_lag_features(
    panel: pd.DataFrame,
    target: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add per-product lag and rolling features, ordered by month.

    All features for a given (product, month) row are computed exclusively
    from observations that occur strictly before that month.
    """
    out = (
        panel.sort_values([PRODUCT_COLUMN, MONTH_COLUMN])
        .reset_index(drop=True)
        .copy()
    )
    grouped = out.groupby(PRODUCT_COLUMN)[target]

    out["sales_lag_1"] = grouped.shift(1)
    out["sales_lag_2"] = grouped.shift(2)
    out["sales_lag_3"] = grouped.shift(3)

    # shift(1) first => the rolling window covers only t-3..t-1, never t.
    out["sales_rolling_3"] = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )
    return out
