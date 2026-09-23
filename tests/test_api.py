"""Tests for the Flask API: health, validation, prediction, error handling.

File-dependent tests skip automatically when the Phase-3 checkpoint/results
have not been generated (run ``python scripts/run_torch_training.py``).
"""

import json
import math

import numpy as np
import pytest

from api.app import create_app
from api.model_service import ModelService
from api.schemas import ValidationError, validate_predict_payload
from src.config import FEATURE_COLUMNS, TORCH_METRICS_PATH, TORCH_MODEL_PATH
from src.torch_model import MLPForecaster

# A plausible product-month feature vector (same contract as training).
VALID_FEATURES = {
    "year": 2019,
    "month_num": 9,
    "quarter": 3,
    "avg_price": 380.55,
    "list_price": 400.0,
    "sales_lag_1": 15234.11,
    "sales_lag_2": 14980.44,
    "sales_lag_3": 16001.9,
    "sales_rolling_3": 15405.48,
}

needs_model = pytest.mark.skipif(
    not TORCH_MODEL_PATH.exists(),
    reason="checkpoint not generated (run scripts/run_torch_training.py)",
)
needs_metrics = pytest.mark.skipif(
    not TORCH_METRICS_PATH.exists(),
    reason="pytorch_metrics.json not generated",
)


@pytest.fixture()
def client():
    return create_app().test_client()


def _broken_service(tmp_path) -> ModelService:
    return ModelService(model_path=tmp_path / "missing.pt",
                        metrics_path=tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["checkpoint_exists"], bool)


def test_health_ok_even_without_checkpoint(tmp_path):
    app = create_app(service=_broken_service(tmp_path))
    resp = app.test_client().get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["checkpoint_exists"] is False


# ---------------------------------------------------------------------------
# GET /model
# ---------------------------------------------------------------------------
@needs_model
@needs_metrics
def test_model_returns_features_and_real_metrics(client):
    resp = client.get("/model")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["model"] == "pytorch_mlp"
    # Same feature contract the model was trained with.
    assert body["features"] == list(FEATURE_COLUMNS)
    # Metrics are loaded from the results file, never hardcoded.
    m = body["holdout_metrics"]
    assert m is not None
    for key in ("mae", "rmse", "mape_percent", "r2"):
        assert m[key] is None or math.isfinite(m[key])
    assert m["mae"] >= 0 and m["rmse"] >= 0
    assert body["training"] is not None


def test_model_503_when_checkpoint_missing(tmp_path):
    app = create_app(service=_broken_service(tmp_path))
    resp = app.test_client().get("/model")
    assert resp.status_code == 503
    assert resp.get_json()["error"]["code"] == "model_unavailable"


# ---------------------------------------------------------------------------
# POST /predict — success
# ---------------------------------------------------------------------------
@needs_model
def test_predict_success(client):
    resp = client.post("/predict", json={"features": VALID_FEATURES})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["model"] == "pytorch_mlp"
    assert isinstance(body["prediction"], float)
    assert math.isfinite(body["prediction"])
    # The API echoes exactly the features it scored (JSON objects are
    # unordered, so compare as sets).
    assert set(body["features_used"]) == set(FEATURE_COLUMNS)
    assert body["checkpoint"].endswith("pytorch_model.pt")


@needs_model
def test_predict_matches_direct_model_call():
    """The endpoint must return exactly what the loaded model predicts."""
    x = np.array([[VALID_FEATURES[name] for name in FEATURE_COLUMNS]])
    direct = float(MLPForecaster.load(TORCH_MODEL_PATH).predict(x)[0])
    resp = create_app().test_client().post("/predict", json={"features": VALID_FEATURES})
    assert resp.status_code == 200
    # Endpoint rounds to 2 decimals for currency.
    assert resp.get_json()["prediction"] == pytest.approx(round(direct, 2), rel=1e-9)


@needs_model
def test_predict_is_deterministic(client):
    a = client.post("/predict", json={"features": VALID_FEATURES}).get_json()
    b = client.post("/predict", json={"features": VALID_FEATURES}).get_json()
    assert a["prediction"] == b["prediction"]


# ---------------------------------------------------------------------------
# POST /predict — validation errors (4xx)
# ---------------------------------------------------------------------------
def test_missing_features_field(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    err = resp.get_json()["error"]
    assert err["code"] == "validation_error"
    assert "features" in err["details"][0]


def test_missing_single_feature_lists_it(client):
    partial = dict(VALID_FEATURES)
    del partial["sales_rolling_3"]
    resp = client.post("/predict", json={"features": partial})
    assert resp.status_code == 400
    assert "sales_rolling_3" in resp.get_json()["error"]["details"][0]


def test_unknown_feature_rejected(client):
    payload = {"features": dict(VALID_FEATURES, sales_lag_99=1.0)}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert "sales_lag_99" in resp.get_json()["error"]["details"][0]


def test_non_numeric_feature_rejected(client):
    payload = {"features": dict(VALID_FEATURES, sales_lag_1="15000")}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert "must be a number" in resp.get_json()["error"]["details"][0]


def test_boolean_feature_rejected(client):
    payload = {"features": dict(VALID_FEATURES, year=True)}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400


def test_non_finite_feature_rejected(client):
    payload = {"features": dict(VALID_FEATURES, sales_lag_1=float("nan"))}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert "finite" in resp.get_json()["error"]["details"][0]


def test_features_wrong_type_rejected(client):
    resp = client.post("/predict", json={"features": [1, 2, 3]})
    assert resp.status_code == 400


def test_wrong_content_type_returns_415(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 415
    assert resp.get_json()["error"]["code"] == "unsupported_media_type"


def test_malformed_json_returns_400(client):
    resp = client.post("/predict", data="{not json", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_json"


def test_predict_503_when_checkpoint_missing(tmp_path):
    """Missing checkpoint surfaces as 503, not a crash."""
    app = create_app(service=_broken_service(tmp_path))
    resp = app.test_client().post("/predict", json={"features": VALID_FEATURES})
    assert resp.status_code == 503
    assert resp.get_json()["error"]["code"] == "model_unavailable"


# ---------------------------------------------------------------------------
# Schema unit tests
# ---------------------------------------------------------------------------
def test_validate_returns_ordered_floats():
    out = validate_predict_payload({"features": VALID_FEATURES})
    assert list(out) == list(FEATURE_COLUMNS)
    assert all(isinstance(v, float) for v in out.values())


def test_validate_collects_multiple_errors():
    partial = {k: v for k, v in VALID_FEATURES.items() if k != "quarter"}
    with pytest.raises(ValidationError) as exc:
        validate_predict_payload(
            {"features": dict(partial, bogus=1, avg_price="cheap")}
        )
    details = "\n".join(exc.value.details)
    assert "missing features: quarter" in details
    assert "unknown features: bogus" in details
    assert "avg_price" in details


def test_validate_rejects_non_dict_body():
    with pytest.raises(ValidationError):
        validate_predict_payload(["not", "a", "dict"])
