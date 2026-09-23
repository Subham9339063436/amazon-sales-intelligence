"""Request validation for the Flask API.

Deliberately dependency-free: plain Python validation against the *same*
``FEATURE_COLUMNS`` contract the model was trained with (src.config), so the
API can never accept a feature vector that differs from training.
"""

from __future__ import annotations

import math

from src.config import FEATURE_COLUMNS

REQUIRED_FEATURES = tuple(FEATURE_COLUMNS)


class ValidationError(Exception):
    """Raised when a request payload fails validation.

    ``details`` is a list of human-readable problem descriptions so the
    endpoint can return every problem at once (HTTP 400).
    """

    def __init__(self, details: list[str]) -> None:
        self.details = list(details)
        super().__init__("; ".join(self.details))


def validate_predict_payload(payload: object) -> dict[str, float]:
    """Validate a ``POST /predict`` body and return an ordered feature dict.

    Expected shape::

        {"features": {"year": 2019, "month_num": 9, ...}}

    Rules:
      * body must be a JSON object with a ``features`` object,
      * every feature in ``FEATURE_COLUMNS`` must be present (none missing),
      * no unknown feature names (typos must fail loudly),
      * every value must be a finite JSON number (bools rejected).

    Raises
    ------
    ValidationError
        With one message per problem found.
    """
    details: list[str] = []

    if not isinstance(payload, dict):
        raise ValidationError(["request body must be a JSON object"])

    if "features" not in payload:
        raise ValidationError(
            [
                "missing required top-level field 'features' "
                "(an object mapping feature names to numbers)"
            ]
        )

    features = payload["features"]
    if not isinstance(features, dict):
        raise ValidationError(["'features' must be a JSON object mapping names to numbers"])

    provided = set(features)
    required = set(REQUIRED_FEATURES)

    missing = sorted(required - provided)
    if missing:
        details.append(f"missing features: {', '.join(missing)}")

    unknown = sorted(provided - required)
    if unknown:
        details.append(
            f"unknown features: {', '.join(unknown)} (allowed: {', '.join(REQUIRED_FEATURES)})"
        )

    clean: dict[str, float] = {}
    for name in REQUIRED_FEATURES:
        if name not in features:
            continue
        value = features[name]
        # bool is a subclass of int in Python — reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            details.append(
                f"feature '{name}' must be a number, got {type(value).__name__}"
            )
            continue
        if not math.isfinite(value):
            details.append(f"feature '{name}' must be a finite number, got {value!r}")
            continue
        clean[name] = float(value)

    if details:
        raise ValidationError(details)

    # Return in the canonical training order.
    return {name: clean[name] for name in REQUIRED_FEATURES}
