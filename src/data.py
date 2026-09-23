"""Data loading and aggregation into the monthly modeling panel."""

from pathlib import Path

import pandas as pd

from .config import RAW_DATA_PATH


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle invoice-line CSV and parse the invoice date."""
    df = pd.read_csv(path)
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"])
    return df


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame whose ``Invoice Date`` column is datetime-typed."""
    if pd.api.types.is_datetime64_any_dtype(df["Invoice Date"]):
        return df
    out = df.copy()
    out["Invoice Date"] = pd.to_datetime(out["Invoice Date"])
    return out


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate invoice lines to one row per product-month.

    Mirrors the aggregation of the original notebook. Note: ``groupby``
    drops rows with a missing ``Item Class`` by default, exactly as the
    original notebook did.
    """
    df = _ensure_datetime(df)
    month_key = df["Invoice Date"].dt.to_period("M").rename("month_period")
    panel = (
        df.groupby([month_key, "Item", "Item Class"])
        .agg(
            sales_amount=("Sales Amount", "sum"),
            quantity=("Sales Quantity", "sum"),
            avg_price=("Sales Price", "mean"),
            list_price=("List Price", "mean"),
            discount=("Discount Amount", "sum"),
            margin=("Sales Margin Amount", "sum"),
        )
        .reset_index()
        .rename(columns={"Item": "product", "Item Class": "category"})
    )
    panel["month"] = panel["month_period"].dt.to_timestamp()
    panel = panel.drop(columns=["month_period"])
    panel["year"] = panel["month"].dt.year
    panel["month_num"] = panel["month"].dt.month
    panel["quarter"] = panel["month"].dt.quarter
    panel = panel.sort_values(["product", "month"]).reset_index(drop=True)
    return panel


def build_total_monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all invoice lines to one total-revenue row per month.

    Returns a Prophet-compatible frame with columns ``ds`` (month start) and
    ``y`` (total sales amount).
    """
    df = _ensure_datetime(df)
    monthly = (
        df.groupby(df["Invoice Date"].dt.to_period("M"))["Sales Amount"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["month_period", "y"]
    out = pd.DataFrame(
        {
            "ds": monthly["month_period"].dt.to_timestamp(),
            "y": monthly["y"].astype(float),
        }
    )
    return out.sort_values("ds").reset_index(drop=True)
