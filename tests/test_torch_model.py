"""Tests for the Phase-3 PyTorch forecaster: loading, shapes, metrics.

File-contract tests skip when the Phase-3 results have not been generated
(run ``python scripts/run_torch_training.py`` first).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    PREDICTIONS_PATH,
    TORCH_METRICS_PATH,
    TORCH_MODEL_PATH,
    TORCH_PREDICTIONS_PATH,
)
from src.metrics import regression_metrics
from src.torch_model import MLPForecaster, set_seed


# ---------------------------------------------------------------------------
# Unit-level: model behaviour (always run)
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_data():
    """Small deterministic regression problem."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURE_COLUMNS)))
    y = 3.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=200) * 100 + 500
    return X[:160], y[:160], X[160:], y[160:]


def test_predict_shape(toy_data):
    X_tr, y_tr, X_te, _ = toy_data
    model = MLPForecaster(seed=42, max_epochs=30, patience=30)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_te),)  # exactly 1-D, one per row
    assert np.isfinite(preds).all()


def test_save_load_roundtrip_identical_predictions(tmp_path, toy_data):
    """Model loading: a saved+reloaded model must reproduce predictions."""
    X_tr, y_tr, X_te, _ = toy_data
    model = MLPForecaster(seed=42, max_epochs=30, patience=30)
    model.fit(X_tr, y_tr)

    path = tmp_path / "model.pt"
    model.save(path)
    assert path.exists()

    reloaded = MLPForecaster.load(path)
    assert reloaded.is_fitted
    assert reloaded.x_mean.shape == (len(FEATURE_COLUMNS),)

    original = model.predict(X_te)
    restored = reloaded.predict(X_te)
    np.testing.assert_allclose(restored, original, rtol=1e-5, atol=1e-3)


def test_scaler_statistics_fit_on_training_only(toy_data):
    """x_mean/std must equal the training partition's, not the full data."""
    X_tr, y_tr, X_te, _ = toy_data
    model = MLPForecaster(seed=42, max_epochs=10, patience=10)
    model.fit(X_tr, y_tr)
    np.testing.assert_allclose(model.x_mean, X_tr.mean(axis=0), rtol=1e-6)
    assert model.y_mean == pytest.approx(float(y_tr.mean()))
    # And they must differ from stats over combined data (else leakage).
    X_all = np.vstack([X_tr, X_te])
    assert not np.allclose(model.x_mean, X_all.mean(axis=0))


def test_fit_rejects_mismatched_lengths(toy_data):
    X_tr, y_tr, _, _ = toy_data
    model = MLPForecaster(seed=0, max_epochs=5)
    with pytest.raises(ValueError):
        model.fit(X_tr, y_tr[:-1])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        MLPForecaster().predict(np.zeros((3, len(FEATURE_COLUMNS))))


def test_metric_calculation_on_known_values():
    """Metric calculation: MAE/RMSE hand-checkable on a tiny vector."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 31.0, 40.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx((2 + 2 + 1 + 0) / 4)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 1 + 0) / 4))
    # MAPE = mean(|e|/y) = (0.2 + 0.1 + 1/30 + 0) / 4 * 100
    assert m["mape_percent"] == pytest.approx(
        (0.2 + 0.1 + 1 / 30 + 0.0) / 4 * 100
    )


def test_training_is_reproducible_with_seed(toy_data):
    X_tr, y_tr, X_te, _ = toy_data
    p1 = MLPForecaster(seed=7, max_epochs=25, patience=25).fit(X_tr, y_tr).predict(X_te)
    p2 = MLPForecaster(seed=7, max_epochs=25, patience=25).fit(X_tr, y_tr).predict(X_te)
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-3)


def test_set_seed_is_idempotent():
    set_seed(123)
    a = np.random.rand()
    set_seed(123)
    b = np.random.rand()
    assert a == b


# ---------------------------------------------------------------------------
# File-contract tests (require generated Phase-3 artifacts)
# ---------------------------------------------------------------------------
needs_results = pytest.mark.skipif(
    not (TORCH_METRICS_PATH.exists() and TORCH_PREDICTIONS_PATH.exists()),
    reason="Phase-3 results not generated yet",
)


@needs_results
def test_saved_model_file_loads():
    """The checkpoint written by the training script must load cleanly."""
    assert TORCH_MODEL_PATH.exists(), "run scripts/run_torch_training.py"
    model = MLPForecaster.load(TORCH_MODEL_PATH)
    assert model.is_fitted
    preds = model.predict(np.zeros((5, len(FEATURE_COLUMNS))))
    assert preds.shape == (5,)


@needs_results
def test_pytorch_predictions_file_contract():
    preds = pd.read_csv(TORCH_PREDICTIONS_PATH)
    assert set(preds.columns) == {"task", "period", "product", "y_true", "model", "y_pred"}
    assert set(preds["model"].unique()) == {"pytorch_mlp"}
    assert set(preds["task"].unique()) == {"panel_product_month"}
    assert preds["y_true"].notna().all() and preds["y_pred"].notna().all()
    assert preds["y_pred"].dtype.kind == "f"
    # Same holdout rows as the Phase-1 panel predictions (if present).
    if PREDICTIONS_PATH.exists():
        phase1 = pd.read_csv(PREDICTIONS_PATH)
        panel1 = phase1[phase1["task"] == "panel_product_month"]
        assert set(preds["period"]) == set(panel1["period"])
        assert len(preds) == len(
            panel1[panel1["model"] == "lightgbm"]
        ), "PyTorch must cover the identical holdout rows"


@needs_results
def test_pytorch_metrics_json_contract():
    data = json.loads(Path(TORCH_METRICS_PATH).read_text())
    for key in ("generated_at_utc", "model", "split", "training", "metrics",
                "baseline_comparison", "notes"):
        assert key in data, f"missing key: {key}"
    m = data["metrics"]
    for metric in ("mae", "rmse", "mape_percent", "r2"):
        assert metric in m
        assert m[metric] is None or np.isfinite(m[metric])
    assert m["mae"] >= 0 and m["rmse"] >= 0
    # Early stopping must have used training-tail validation only.
    assert data["training"]["early_stopping"] is True
    assert data["training"]["n_val_rows"] > 0
    assert data["training"]["n_val_rows"] < data["training"]["n_train_rows"]


@needs_results
def test_metrics_match_predictions_recomputed_mae():
    """The stored MAE must equal the MAE recomputed from the predictions CSV."""
    data = json.loads(Path(TORCH_METRICS_PATH).read_text())
    preds = pd.read_csv(TORCH_PREDICTIONS_PATH)
    recomputed = (preds["y_true"] - preds["y_pred"]).abs().mean()
    assert recomputed == pytest.approx(data["metrics"]["mae"], rel=1e-6)


@needs_results
def test_baseline_comparison_uses_same_test_period():
    """Phase-3 comparison must reference the identical holdout months."""
    torch_result = json.loads(Path(TORCH_METRICS_PATH).read_text())
    comp = torch_result["baseline_comparison"]
    assert comp["available"] is True, "Phase-1 metrics file missing"
    assert comp["same_test_period"] is True
    assert comp["phase1_test_month_range"] == comp["pytorch_test_month_range"]
    # Baseline metrics must have been loaded from the Phase-1 file.
    assert METRICS_PATH.exists()
    phase1 = json.loads(Path(METRICS_PATH).read_text())
    phase1_mae = phase1["holdout_panel_product_month"]["metrics"]["lightgbm"]["mae"]
    assert comp["baseline_holdout_metrics"]["lightgbm"]["mae"] == phase1_mae
