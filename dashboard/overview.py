"""Executive overview: headline KPIs and revenue trend."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(df: pd.DataFrame) -> None:
    """KPI cards + monthly revenue trend for the filtered sales frame."""
    st.subheader("Executive Overview")

    if df.empty:
        st.info("No rows match the current filters.")
        return

    total_revenue = df["Sales Amount"].sum()
    n_orders = df["Order Number"].nunique()
    n_customers = df["Custkey"].nunique()
    n_products = df["Item"].nunique()
    total_units = df["Sales Quantity"].sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total revenue", f"${total_revenue:,.0f}")
    k2.metric("Orders", f"{n_orders:,}")
    k3.metric("Customers", f"{n_customers:,}")
    k4.metric("Products sold", f"{n_products:,}")
    k5.metric("Units", f"{total_units:,}")

    st.caption(
        f"{len(df):,} invoice lines · "
        f"{df['Invoice Date'].min().date()} → {df['Invoice Date'].max().date()}"
    )

    st.markdown("##### Revenue trend (monthly)")
    trend = (
        df.groupby("month")
        .agg(revenue=("Sales Amount", "sum"), orders=("Order Number", "nunique"))
        .reset_index()
        .sort_values("month")
    )
    chart_df = trend.set_index("month")[["revenue"]]
    st.line_chart(chart_df, y_label="Revenue ($)", height=300)

    with st.expander("Monthly revenue table"):
        show = trend.copy()
        show["revenue"] = show["revenue"].round(2)
        show.columns = ["Month", "Revenue ($)", "Orders"]
        st.dataframe(show, use_container_width=True, hide_index=True)
