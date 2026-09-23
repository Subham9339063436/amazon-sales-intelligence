"""Sales analysis: monthly revenue, category/product performance, seasonality."""

from __future__ import annotations

import pandas as pd
import streamlit as st

_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def render(df: pd.DataFrame) -> None:
    """Three sections built only from real invoice-line columns."""
    st.subheader("Sales Analysis")

    if df.empty:
        st.info("No rows match the current filters.")
        return

    # --- Monthly revenue -------------------------------------------------
    st.markdown("##### Monthly revenue")
    monthly = (
        df.groupby("month")["Sales Amount"].sum().reset_index().sort_values("month")
    )
    st.bar_chart(monthly.set_index("month"), y="Sales Amount", height=300)

    # --- Category / product performance ----------------------------------
    st.markdown("##### Category & product performance")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("Revenue by item class")
        by_class = (
            df.groupby("Item Class")["Sales Amount"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        st.bar_chart(by_class.set_index("Item Class"), y="Sales Amount", height=300)

    with c2:
        st.caption("Top 10 products by revenue")
        top = (
            df.groupby("Item")["Sales Amount"]
            .sum()
            .nlargest(10)
            .reset_index()
            .sort_values("Sales Amount")
        )
        st.bar_chart(top.set_index("Item"), y="Sales Amount", height=300)

    # --- Seasonal trends -------------------------------------------------
    st.markdown("##### Seasonal trends")
    s1, s2, s3 = st.columns(3)

    with s1:
        st.caption("Average revenue by calendar month (all years)")
        by_month = df.groupby(df["Invoice Date"].dt.month)["Sales Amount"].mean()
        st.bar_chart(by_month, y_label="Avg revenue ($)", height=280)

    with s2:
        st.caption("Revenue by quarter")
        by_q = df.groupby("quarter")["Sales Amount"].sum()
        st.bar_chart(by_q, y_label="Revenue ($)", height=280)

    with s3:
        st.caption("Average revenue by day of week")
        by_dow = df.groupby("day_of_week")["Sales Amount"].mean()
        by_dow.index = [_DOW_NAMES[i] for i in by_dow.index]
        st.bar_chart(by_dow, y_label="Avg revenue ($)", height=280)

    st.caption(
        "Data note: 2018-04 → 2018-12 has no observations in the source data; "
        "monthly/seasonal charts reflect observed months only."
    )
