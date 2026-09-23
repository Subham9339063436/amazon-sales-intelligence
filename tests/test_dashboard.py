"""Tests for the Streamlit dashboard: app boot + results-file contracts.

These use Streamlit's AppTest to execute app.py headlessly, which catches
import errors, missing files, and rendering exceptions without a browser.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import METRICS_PATH, PREDICTIONS_PATH, PROJECT_ROOT

pytestmark = pytest.mark.skipif(
    not METRICS_PATH.exists(),
    reason="results/model_metrics.json not generated yet",
)


def test_app_renders_without_exception():
    """app.py must execute all four tabs with no runtime error."""
    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert not at.exception, f"App raised: {at.exception}"
    # Title + all four tabs rendered.
    assert len(at.tabs) == 4, f"expected 4 tabs, got {len(at.tabs)}"
    assert any("Amazon Sales" in t.value for t in at.title)


def test_app_survives_narrow_filter_selection():
    """Mid-selection date_input (1-tuple) must not crash the app."""
    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    # Simulate a deselected end-date (Streamlit then yields a 1-tuple).
    at.date_input[0].set_value(at.date_input[0].value[:1]).run()
    assert not at.exception, f"App raised on partial date range: {at.exception}"


def test_metrics_file_contains_required_sections():
    """The forecasting tab depends on these exact keys."""
    data = json.loads(Path(METRICS_PATH).read_text())
    for key in ("split", "walk_forward_cv", "holdout_panel_product_month",
                "holdout_revenue_monthly_total", "generated_at_utc", "notes"):
        assert key in data, f"missing key: {key}"

    for block in (data["holdout_panel_product_month"]["metrics"],
                  data["holdout_revenue_monthly_total"]["metrics"]):
        for model, m in block.items():
            for metric in ("mae", "rmse", "mape_percent", "r2"):
                assert metric in m, f"{model} missing {metric}"


def test_predictions_file_contract():
    """The actual-vs-predicted charts depend on this exact schema."""
    preds = pd.read_csv(PREDICTIONS_PATH)
    assert set(preds.columns) == {"task", "period", "product", "y_true", "model", "y_pred"}
    assert set(preds["task"].unique()) == {"panel_product_month", "revenue_monthly_total"}
    assert preds["y_true"].notna().all() and preds["y_pred"].notna().all()
    # Revenue rows: both models present for every holdout month.
    rev = preds[preds["task"] == "revenue_monthly_total"]
    assert set(rev["model"].unique()) == {"naive_last_value", "prophet"}
    assert rev.groupby("period")["model"].nunique().eq(2).all()
    # Panel rows: 3 models over the 5 holdout months.
    panel = preds[preds["task"] == "panel_product_month"]
    assert set(panel["model"].unique()) == {"naive_last_value", "linear_regression", "lightgbm"}
    assert panel["period"].nunique() == 5


def test_metrics_match_predictions_consistency():
    """MAE recomputed from predictions must equal the stored JSON MAE."""
    data = json.loads(Path(METRICS_PATH).read_text())
    preds = pd.read_csv(PREDICTIONS_PATH)

    checks = [
        ("panel_product_month", "lightgbm",
         data["holdout_panel_product_month"]["metrics"]["lightgbm"]["mae"]),
        ("revenue_monthly_total", "prophet",
         data["holdout_revenue_monthly_total"]["metrics"]["prophet"]["mae"]),
    ]
    for task, model, expected_mae in checks:
        subset = preds[(preds["task"] == task) & (preds["model"] == model)]
        recomputed = (subset["y_true"] - subset["y_pred"]).abs().mean()
        assert recomputed == pytest.approx(expected_mae, rel=1e-6), (
            f"MAE mismatch for {task}/{model}: {recomputed} vs {expected_mae}"
        )
