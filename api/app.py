"""Flask REST API for the Phase-3 PyTorch sales forecaster.

Endpoints
---------
  GET  /health   — liveness + model status (never requires the model)
  GET  /model    — model metadata and *actual* holdout metrics from
                   results/pytorch_metrics.json
  POST /predict  — predict product-month sales from engineered features

Run from the repository root (pick one):

    python api/app.py                     # dev server on $PORT (default 5001)
    gunicorn -b 0.0.0.0:5001 -w 1 api.app:app   # production-style

Example request:

    curl -s -X POST http://localhost:5001/predict \
      -H 'Content-Type: application/json' \
      -d '{"features": {
            "year": 2019, "month_num": 9, "quarter": 3,
            "avg_price": 380.55, "list_price": 400.0,
            "sales_lag_1": 15234.11, "sales_lag_2": 14980.44,
            "sales_lag_3": 16001.90, "sales_rolling_3": 15405.48}}'

Example response:

    {"prediction": 15712.34, "target": "product-month sales_amount (USD)",
     "model": "pytorch_mlp", "checkpoint": "models/pytorch_model.pt",
     "features_used": {...}}

Validation failures return HTTP 400 with every problem listed under
``error.details``; a missing/broken checkpoint returns HTTP 503.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `python api/app.py` from the repository root: put the root on sys.path
# so both `api.*` and `src.*` import as packages.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request  # noqa: E402

from api.model_service import ModelNotAvailableError, ModelService  # noqa: E402
from api.schemas import ValidationError, validate_predict_payload  # noqa: E402

SERVICE_UNAVAILABLE = (
    {
        "error": {
            "code": "model_unavailable",
            "message": (
                "The model checkpoint is missing or could not be loaded. "
                "Run: python scripts/run_torch_training.py"
            ),
        }
    },
    503,
)


def create_app(service: ModelService | None = None) -> Flask:
    """Application factory (injectable service keeps tests hermetic)."""
    svc = service if service is not None else ModelService()
    app = Flask(__name__)

    @app.get("/health")
    def health():
        """Liveness probe — always 200; reports model state separately."""
        return jsonify(
            {
                "status": "ok",
                "service": "amazon-sales-forecasting-api",
                "model_loaded": svc.loaded,
                "checkpoint_exists": svc.checkpoint_exists,
            }
        )

    @app.get("/model")
    def model_info():
        """Model metadata + real holdout metrics loaded from the results file."""
        if not svc.checkpoint_exists:
            return jsonify(SERVICE_UNAVAILABLE[0]), SERVICE_UNAVAILABLE[1]
        try:
            svc.load()
        except ModelNotAvailableError:
            return jsonify(SERVICE_UNAVAILABLE[0]), SERVICE_UNAVAILABLE[1]
        return jsonify(svc.metadata())

    @app.post("/predict")
    def predict():
        if not request.is_json:
            return (
                jsonify(
                    {
                        "error": {
                            "code": "unsupported_media_type",
                            "message": "Send Content-Type: application/json",
                        }
                    }
                ),
                415,
            )

        payload = request.get_json(silent=True)
        if payload is None:
            return (
                jsonify(
                    {
                        "error": {
                            "code": "invalid_json",
                            "message": "Request body is not valid JSON",
                        }
                    }
                ),
                400,
            )

        try:
            features = validate_predict_payload(payload)
        except ValidationError as exc:
            return (
                jsonify(
                    {
                        "error": {
                            "code": "validation_error",
                            "details": exc.details,
                        }
                    }
                ),
                400,
            )

        try:
            value = svc.predict(features)
        except ModelNotAvailableError as exc:
            return jsonify({"error": {"code": "model_unavailable", "message": str(exc)}}), 503

        return jsonify(
            {
                "prediction": value,
                "target": "product-month sales_amount (USD)",
                "model": "pytorch_mlp",
                "checkpoint": svc.metadata()["checkpoint"],
                "features_used": features,
            }
        )

    @app.errorhandler(404)
    @app.errorhandler(405)
    def _json_error(error):
        """JSON bodies for routing errors too, so clients never get HTML."""
        code = "not_found" if error.code == 404 else "method_not_allowed"
        return jsonify({"error": {"code": code, "message": error.description}}), error.code

    return app


# WSGI entry point (gunicorn `api.app:app`) and `python api/app.py`.
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
