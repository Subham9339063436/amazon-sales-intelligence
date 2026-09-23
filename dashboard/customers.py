"""Customer analysis: Phase-1 clusters + RFM-style metrics from real columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data import RFM_COLUMNS


def _cluster_summary(seg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate real per-customer columns for each cluster."""
    summary = seg.groupby("cluster").agg(
        customers=("customer_id", "count"),
        avg_total_spent=("total_spent", "mean"),
        median_total_spent=("total_spent", "median"),
        avg_order_value=("avg_order_value", "mean"),
        avg_order_frequency=("order_frequency", "mean"),
        avg_days_since_purchase=("days_since_last_purchase", "mean"),
        avg_profit_margin=("profit_margin", "mean"),
        pct_over_90d_no_purchase=(
            "days_since_last_purchase",
            lambda s: float(np.mean(s.to_numpy() > 90) * 100),
        ),
    )
    return summary.round(2)


def render(seg: pd.DataFrame, clusters: list[int], activity: pd.DataFrame) -> None:
    """Cluster sizes, RFM views and per-cluster metrics.

    Parameters
    ----------
    seg : customer_segments.csv (all customers, with cluster labels)
    clusters : cluster ids selected in the sidebar
    activity : filtered raw sales frame (for the in-filter customer count)
    """
    st.subheader("Customer Analysis")

    selected = seg[seg["cluster"].isin(clusters)]
    if selected.empty:
        st.info("No customers in the selected clusters.")
        return

    rec, freq, mon = (
        RFM_COLUMNS["recency"],
        RFM_COLUMNS["frequency"],
        RFM_COLUMNS["monetary"],
    )

    # --- KPI cards -------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers (segments file)", f"{len(seg):,}")
    k2.metric("Customers in selection", f"{len(selected):,}")
    k3.metric("Clusters", f"{selected['cluster'].nunique()}")
    k4.metric(
        "Customers in filtered sales",
        f"{activity['Custkey'].nunique():,}" if not activity.empty else "0",
    )

    # --- Cluster sizes ---------------------------------------------------
    st.markdown("##### Cluster sizes")
    sizes = selected["cluster"].value_counts().sort_index()
    st.bar_chart(sizes, y_label="Customers", height=280)

    # --- RFM-style views -------------------------------------------------
    st.markdown("##### RFM views (from `customer_segments.csv`)")
    r1, r2 = st.columns(2)
    with r1:
        st.caption("Recency vs Monetary, colored by cluster")
        st.scatter_chart(
            selected,
            x=rec,
            y=mon,
            color="cluster",
            height=300,
        )
    with r2:
        st.caption("Frequency vs Monetary, colored by cluster")
        st.scatter_chart(
            selected,
            x=freq,
            y=mon,
            color="cluster",
            height=300,
        )
    st.caption(
        "R = days_since_last_purchase, F = order_frequency (orders), "
        "M = total_spent. These are the same per-customer metrics the "
        "notebook used to fit KMeans."
    )

    # --- Per-cluster metrics ---------------------------------------------
    st.markdown("##### Cluster metrics")
    st.dataframe(_cluster_summary(selected), use_container_width=True)

    # --- Highest-value customers in selection -----------------------------
    st.markdown("##### Top 10 customers by total spend (selection)")
    cols = [
        "customer_id",
        "cluster",
        "total_spent",
        "avg_order_value",
        "order_frequency",
        "days_since_last_purchase",
        "profit_margin",
    ]
    top = (
        selected.nlargest(10, "total_spent")[cols]
        .rename(
            columns={
                "customer_id": "Customer",
                "total_spent": "Total spent ($)",
                "avg_order_value": "AOV ($)",
                "order_frequency": "Orders",
                "days_since_last_purchase": "Days since last",
                "profit_margin": "Margin",
            }
        )
        .round(2)
    )
    st.dataframe(top, use_container_width=True, hide_index=True)
