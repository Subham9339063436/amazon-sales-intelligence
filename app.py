"""Streamlit entry point: Amazon Sales Forecasting & Analytics dashboard.

Run from the repository root:

    streamlit run app.py

Tabs: Executive Overview · Sales Analysis · Customer Analysis · Forecasting.
Sales tabs react to the sidebar filters; metrics on the Forecasting tab are
loaded from results/model_metrics.json + results/holdout_predictions.csv
(regenerate with ``python scripts/run_training.py``).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Amazon Sales Analytics",
    page_icon="📊",
    layout="wide",
)

from dashboard import customers, forecasting, overview, sales
from dashboard.data import get_sales, get_segments
from dashboard.filters import apply_sales_filters, render_cluster_filter, render_sales_filters

# ---------------------------------------------------------------- load data
sales_df = get_sales()
segments_df = get_segments()

# ------------------------------------------------------------------ filters
filters = render_sales_filters(sales_df)
clusters = render_cluster_filter(segments_df)
filtered_sales = apply_sales_filters(sales_df, filters)

active = []
if filters.start_date != sales_df["Invoice Date"].min().date() or (
    filters.end_date != sales_df["Invoice Date"].max().date()
):
    active.append(f"date {filters.start_date} → {filters.end_date}")
if len(clusters) < segments_df["cluster"].nunique():
    active.append(f"clusters {clusters}")
n_cat = sales_df["Item Class"].nunique()
if len(filters.categories) < n_cat:
    active.append(f"{len(filters.categories)}/{n_cat} categories")
if filters.product != "All products":
    active.append(f"product: {filters.product}")

st.title("📊 Amazon Sales Forecasting & Analytics")
st.caption(
    "Data: Amazon food-category invoice lines (2017-01 → 2019-12) · "
    "segments from KMeans · model metrics from Phase-1 results files"
    + (f" · filters active: {'; '.join(active)}" if active else " · no filters active")
)

tab_overview, tab_sales, tab_customers, tab_forecast = st.tabs(
    ["🏢 Executive Overview", "📈 Sales Analysis", "👥 Customer Analysis", "🔮 Forecasting"]
)

with tab_overview:
    overview.render(filtered_sales)

with tab_sales:
    sales.render(filtered_sales)

with tab_customers:
    customers.render(segments_df, clusters, filtered_sales)

with tab_forecast:
    forecasting.render()
