"""Shared test fixtures: a tiny deterministic product-month panel."""

import numpy as np
import pandas as pd
import pytest

from src.data import build_monthly_panel
from src.features import add_lag_features


@pytest.fixture
def tiny_raw() -> pd.DataFrame:
    """Two products over six months with strictly increasing, distinct sales."""
    rows = []
    months = pd.date_range("2019-01-01", periods=6, freq="MS")
    for p_idx, product in enumerate(["A", "B"]):
        for m_idx, month in enumerate(months):
            rows.append(
                {
                    "Custkey": 1000 + p_idx,
                    "DateKey": month.strftime("%m/%d/%Y"),
                    "Discount Amount": 1.0,
                    "Invoice Date": month.strftime("%Y/%m/%d"),
                    "Invoice Number": 9000 + m_idx,
                    "Item Class": "P01",
                    "Item Number": 100 + p_idx,
                    "Item": product,
                    "Line Number": 1000,
                    "List Price": 50.0,
                    "Order Number": 5000 + m_idx,
                    "Promised Delivery Date": month.strftime("%m/%d/%Y"),
                    # Distinct target values so leakage is detectable.
                    "Sales Amount": float(100 * (m_idx + 1) + p_idx),
                    "Sales Amount Based on List Price": 0.0,
                    "Sales Cost Amount": 10.0,
                    "Sales Margin Amount": 5.0,
                    "Sales Price": 10.0,
                    "Sales Quantity": 1,
                    "Sales Rep": 1,
                    "U/M": "EA",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_panel(tiny_raw: pd.DataFrame) -> pd.DataFrame:
    return build_monthly_panel(tiny_raw)


@pytest.fixture
def tiny_modeling(tiny_panel: pd.DataFrame) -> pd.DataFrame:
    return add_lag_features(tiny_panel)


def make_panel(values_by_product: dict[str, list[float]]) -> pd.DataFrame:
    """Build a panel directly from per-product sales series (month-indexed)."""
    frames = []
    for product, values in values_by_product.items():
        months = pd.date_range("2019-01-01", periods=len(values), freq="MS")
        frames.append(
            pd.DataFrame(
                {
                    "product": product,
                    "month": months,
                    "sales_amount": np.asarray(values, dtype=float),
                    "year": months.year,
                    "month_num": months.month,
                    "quarter": months.quarter,
                    "avg_price": 10.0,
                    "list_price": 50.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
