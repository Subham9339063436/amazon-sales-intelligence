"""Model service: modular, thread-safe loading and inference.

Wraps the Phase-3 ``MLPForecaster`` checkpoint (``models/pytorch_model.pt``)
behind a small lazy-loading interface so the API never duplicates training or
preprocessing logic:

  * features come from the shared ``src.config.FEATURE_COLUMNS`` contract,
  * the model is loaded once, on first use, from the checkpoint written by
    ``scripts/run_torch_training.py``,
  * metadata/metrics are read from ``results/pytorch_metrics.json`` — never
    hardcoded.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from src.config import (
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TORCH_METRICS_PATH,
    TORCH_MODEL_PATH,
)
from src.torch_model import MLPForecaster


class ModelNotAvailableError(RuntimeError):
    """The checkpoint is missing or could not be loaded."""


def _rel(path: Path) -> str:
    """Path relative to the repository root (for stable API responses)."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class ModelService:
    """Lazily loads the checkpoint once and serves predictions from it."""

    def __init__(
        self,
        model_path: Path = TORCH_MODEL_PATH,
        metrics_path: Path = TORCH_METRICS_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self._model: MLPForecaster | None = None
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------
    @property
    def checkpoint_exists(self) -> bool:
        return self.model_path.exists()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> MLPForecaster:
        """Return the model, loading it on first use (thread-safe)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not self.model_path.exists():
                        raise ModelNotAvailableError(
                            f"model checkpoint not found at {_rel(self.model_path)} — "
                            "run: python scripts/run_torch_training.py"
                        )
                    try:
                        self._model = MLPForecaster.load(self.model_path)
                    except Exception as exc:  # corrupt/incompatible file
                        raise ModelNotAvailableError(
                            f"failed to load checkpoint {_rel(self.model_path)}: {exc}"
                        ) from exc
        return self._model

    # -- inference ---------------------------------------------------------
    def predict(self, features: dict[str, float]) -> float:
        """Predict product-month sales for one validated feature vector."""
        model = self.load()
        x = np.array(
            [[features[name] for name in FEATURE_COLUMNS]], dtype=np.float64
        )
        prediction = model.predict(x)
        return round(float(prediction[0]), 2)

    # -- metadata ----------------------------------------------------------
    def metadata(self) -> dict:
        """Model info: structure from config, metrics from the results file."""
        meta: dict = {
            "model": "pytorch_mlp",
            "label": "PyTorch MLP forecaster (Phase 3)",
            "architecture": "MLP (64, 32) ReLU → linear",
            "target": "product-month sales_amount (USD)",
            "checkpoint": _rel(self.model_path),
            "checkpoint_exists": self.checkpoint_exists,
            "features": list(FEATURE_COLUMNS),
            "metrics_file": _rel(self.metrics_path),
            "training": None,
            "split": None,
            "holdout_metrics": None,
            "generated_at_utc": None,
        }
        if self.metrics_path.exists():
            data = json.loads(self.metrics_path.read_text())
            metrics = data.get("metrics") or {}
            meta["holdout_metrics"] = {
                key: metrics.get(key)
                for key in ("mae", "rmse", "mape_percent", "r2")
            }
            meta["training"] = data.get("training")
            meta["split"] = data.get("split")
            meta["generated_at_utc"] = data.get("generated_at_utc")
        return meta
