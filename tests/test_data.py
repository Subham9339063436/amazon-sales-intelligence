"""Tests for data loading and panel aggregation."""

import pandas as pd
import pytest

from src.config import RAW_DATA_PATH
from src.data import build_monthly_panel, build_total_monthly_revenue, load_raw


def test_load_raw_parses_dates():
    if not RAW_DATA_PATH.exists():
        pytest.skip("raw dataset not present")
    df = load_raw()
    assert len(df) == 65280
    assert pd.api.types.is_datetime64_any_dtype(df["Invoice Date"])


def test_panel_aggregation_matches_raw_totals(tiny_raw):
    panel = build_monthly_panel(tiny_raw)
    assert len(panel) == 12  # 2 products x 6 months
    assert panel["sales_amount"].sum() == pytest.approx(
        tiny_raw["Sales Amount"].sum()
    )
    # Time features derived correctly.
    assert set(panel["month_num"]) == set(range(1, 7))
    # Fixture months span Jan..Jun -> quarters 1 and 2.
    assert (panel["quarter"] == panel["month"].dt.quarter).all()
    assert set(panel["quarter"]) == {1, 2}


def test_panel_sorted_by_product_month(tiny_panel):
    ordered = tiny_panel.sort_values(["product", "month"])
    pd.testing.assert_frame_equal(tiny_panel, ordered)


def test_revenue_frame_consistent_with_panel(tiny_raw):
    panel = build_monthly_panel(tiny_raw)
    rev = build_total_monthly_revenue(tiny_raw)
    assert list(rev.columns) == ["ds", "y"]
    assert len(rev) == 6
    # Revenue frame (all rows) >= panel sums (panel drops NaN Item Class rows).
    assert rev["y"].sum() >= panel["sales_amount"].sum() - 1e-9
    # Sorted and monthly frequency.
    assert rev["ds"].is_monotonic_increasing
    assert rev["y"].dtype == float
    assert (rev["y"] > 0).all()  # MAPE stays appropriate for this target
