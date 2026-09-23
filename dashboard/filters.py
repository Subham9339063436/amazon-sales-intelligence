"""Sidebar filters shared across dashboard tabs.

Only real, filterable dimensions of the repository data are offered:
date (Invoice Date), category (Item Class), product (Item) for sales;
cluster/customer for the segments file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st


@dataclass
class SalesFilters:
    start_date: object
    end_date: object
    categories: list[str] = field(default_factory=list)
    product: str = "All products"


def render_sales_filters(df: pd.DataFrame) -> SalesFilters:
    """Date / category / product filters (sidebar). All rows pass by default."""
    st.sidebar.header("Sales filters")

    min_d = df["Invoice Date"].min().date()
    max_d = df["Invoice Date"].max().date()
    selected = st.sidebar.date_input(
        "Invoice date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key="date_range",
    )
    # date_input yields a 1-tuple while the user is mid-selection.
    if selected is not None and len(selected) == 2:
        start, end = selected
    else:
        start, end = min_d, max_d

    categories = sorted(df["Item Class"].unique())
    chosen_cats = st.sidebar.multiselect(
        "Item class (category)",
        options=categories,
        default=categories,
        key="categories",
    )

    products = ["All products"] + sorted(df["Item"].unique())
    product = st.sidebar.selectbox("Product", products, key="product")

    return SalesFilters(
        start_date=start, end_date=end, categories=chosen_cats, product=product
    )


def apply_sales_filters(df: pd.DataFrame, f: SalesFilters) -> pd.DataFrame:
    """Apply the sidebar selections to the sales frame."""
    mask = (
        df["Invoice Date"].dt.date.between(f.start_date, f.end_date)
        & df["Item Class"].isin(f.categories)
    )
    if f.product != "All products":
        mask &= df["Item"] == f.product
    return df[mask]


def render_cluster_filter(seg: pd.DataFrame) -> list[int]:
    """Cluster multi-select (sidebar) used by the customer tab."""
    st.sidebar.header("Customer filters")
    clusters = sorted(seg["cluster"].unique().tolist())
    return st.sidebar.multiselect(
        "Customer clusters",
        options=clusters,
        default=clusters,
        key="clusters",
    )
