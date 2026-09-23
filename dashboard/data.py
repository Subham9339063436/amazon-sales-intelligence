"""Cached data access for the dashboard.

All loaders read *real* repository data:
  - ``data/Amazon_foodcategory_sales.csv`` (sales facts),
  - ``data/customer_segments.csv`` (Phase-1 KMeans output),
  - ``results/model_metrics.json`` (Phase-1 evaluation metrics),
  - ``results/holdout_predictions.csv`` (Phase-1 holdout predictions).

Nothing here fabricates or hardcodes values; missing result files are
surfaced to the caller so the UI can show an actionable message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    METRICS_PATH,
    MODEL_LABELS,  # re-exported for the forecasting tab
    PREDICTIONS_PATH,
    RAW_DATA_PATH,
    SEGMENTS_PATH,
    TORCH_METRICS_PATH,
    TORCH_PREDICTIONS_PATH,
)
from src.data import load_raw

# Columns of customer_segments.csv used for RFM-style views.
RFM_COLUMNS = {
    "recency": "days_since_last_purchase",
    "frequency": "order_frequency",
    "monetary": "total_spent",
}


def results_available() -> dict[str, bool]:
    """Which result files exist (for graceful UI degradation)."""
    return {
        "metrics": METRICS_PATH.exists(),
        "predictions": PREDICTIONS_PATH.exists(),
        "torch_metrics": TORCH_METRICS_PATH.exists(),
        "torch_predictions": TORCH_PREDICTIONS_PATH.exists(),
    }


@st.cache_data
def get_sales() -> pd.DataFrame:
    """Invoice-line sales facts with parsed dates and display helpers."""
    df = load_raw(RAW_DATA_PATH)
    # Keep the 12.7% missing Item Class rows visible as their own bucket
    # instead of silently dropping them.
    df["Item Class"] = df["Item Class"].fillna("(missing)")
    df["year"] = df["Invoice Date"].dt.year
    df["month"] = df["Invoice Date"].dt.to_period("M").dt.to_timestamp()
    df["quarter"] = df["Invoice Date"].dt.quarter
    df["day_of_week"] = df["Invoice Date"].dt.dayofweek
    return df


@st.cache_data
def get_segments() -> pd.DataFrame:
    """customer_segments.csv: per-customer metrics + Phase-1 cluster labels."""
    seg = pd.read_csv(SEGMENTS_PATH)
    seg["cluster"] = seg["cluster"].astype(int)
    return seg


@st.cache_data
def get_metrics() -> dict:
    """Phase-1 model evaluation results (raises FileNotFoundError if absent)."""
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def get_predictions() -> pd.DataFrame:
    """Long-format holdout predictions (raises FileNotFoundError if absent)."""
    preds = pd.read_csv(PREDICTIONS_PATH)
    preds["period"] = preds["period"].astype(str)
    return preds


@st.cache_data
def get_torch_metrics() -> dict:
    """Phase-3 PyTorch evaluation results (raises FileNotFoundError if absent)."""
    with open(TORCH_METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def get_torch_predictions() -> pd.DataFrame:
    """Phase-3 PyTorch holdout predictions (same schema as Phase-1 file)."""
    preds = pd.read_csv(TORCH_PREDICTIONS_PATH)
    preds["period"] = preds["period"].astype(str)
    return preds
