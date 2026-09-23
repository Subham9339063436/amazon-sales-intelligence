"""Train and evaluate the Phase-3 PyTorch forecaster on the Phase-1 holdout.

Guarantees:
  * Same leakage-free features/target as Phase 1 (src.features).
  * Same chronological holdout (src.split.chronological_split, same fraction).
  * Early stopping only on a tail of the *training* period (TORCH_ES_MONTHS),
    mirroring the LightGBM protocol; the test months stay untouched.
  * All metrics computed here from real predictions, then written to
    results/pytorch_metrics.json and results/pytorch_holdout_predictions.csv.
  * The model checkpoint is written to models/pytorch_model.pt.
  * A comparison against the Phase-1 baselines is included by *loading*
    results/model_metrics.json — nothing is hardcoded.

Run from the repository root:
    python scripts/run_torch_training.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_LABELS,
    MONTH_COLUMN,
    PRODUCT_COLUMN,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_FRACTION,
    TORCH_ES_MONTHS,
    TORCH_METRICS_PATH,
    TORCH_MODEL_PATH,
    TORCH_PREDICTIONS_PATH,
)
from src.data import build_monthly_panel, load_raw
from src.features import add_lag_features
from src.metrics import regression_metrics
from src.split import chronological_split
from src.torch_model import MLPForecaster

PYTORCH_MODEL_KEY = "pytorch_mlp"


def load_baseline_comparison(split_info: dict) -> dict:
    """Load Phase-1 panel metrics from disk and verify the test period."""
    if not METRICS_PATH.exists():
        return {
            "source": str(METRICS_PATH.name),
            "available": False,
            "reason": "results/model_metrics.json not found — run scripts/run_training.py first",
        }
    with open(METRICS_PATH) as f:
        phase1 = json.load(f)
    same_period = phase1["split"]["test_month_range"] == split_info["test_month_range"]
    return {
        # Repo-relative so committed results never embed a local machine path.
        "source": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        "available": True,
        "phase1_test_month_range": phase1["split"]["test_month_range"],
        "pytorch_test_month_range": split_info["test_month_range"],
        "same_test_period": bool(same_period),
        # Straight copy of the baseline holdout metrics from the Phase-1 file.
        "baseline_holdout_metrics": phase1["holdout_panel_product_month"]["metrics"],
    }


def main() -> dict:
    print("Loading data (same pipeline as Phase 1) ...")
    raw = load_raw()
    panel = build_monthly_panel(raw)
    modeling = add_lag_features(panel).dropna(subset=FEATURE_COLUMNS)
    modeling = modeling.reset_index(drop=True)

    train, test, split_info = chronological_split(modeling, TEST_FRACTION)
    print(
        f"  train {split_info['train_month_range'][0]}"
        f"..{split_info['train_month_range'][1]} ({split_info['n_train_rows']} rows), "
        f"test {split_info['test_month_range'][0]}"
        f"..{split_info['test_month_range'][1]} ({split_info['n_test_rows']} rows)"
    )

    # Early-stopping tail carved from the *training* period only — the same
    # months LightGBM uses (TORCH_ES_MONTHS == LIGHTGBM_ES_MONTHS).
    train_months = np.sort(train[MONTH_COLUMN].unique())
    n_es = min(TORCH_ES_MONTHS, max(1, len(train_months) // 4))
    es_months = train_months[-n_es:]
    fit_mask = ~train[MONTH_COLUMN].isin(es_months)
    es_frame = train[train[MONTH_COLUMN].isin(es_months)]
    print(
        f"  fit rows: {int(fit_mask.sum())}, "
        f"early-stop rows: {len(es_frame)} "
        f"({pd.Timestamp(es_months[0]).strftime('%Y-%m')}"
        f"..{pd.Timestamp(es_months[-1]).strftime('%Y-%m')})"
    )

    X_fit = train.loc[fit_mask, FEATURE_COLUMNS]
    y_fit = train.loc[fit_mask, TARGET_COLUMN]
    X_es = es_frame[FEATURE_COLUMNS]
    y_es = es_frame[TARGET_COLUMN]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    print("\nTraining PyTorch MLP ...")
    model = MLPForecaster(seed=RANDOM_SEED)
    model.fit(X_fit, y_fit, X_val=X_es, y_val=y_es)
    info = model.training_info
    print(
        f"  epochs_run={info['epochs_run']} best_epoch={info['best_epoch']} "
        f"best_val_mse(norm)={info['best_monitor']:.6f}"
    )

    preds = model.predict(X_test)
    metrics = regression_metrics(y_test.to_numpy(), preds)
    print(
        f"\nHoldout (panel: product-month sales, {split_info['test_month_range'][0]}"
        f"..{split_info['test_month_range'][1]}):"
    )
    print(
        f"  {PYTORCH_MODEL_KEY:<22} MAE={metrics['mae']:>12,.2f}  "
        f"RMSE={metrics['rmse']:>12,.2f}  MAPE={metrics['mape_percent']:.2f}%  "
        f"R2={metrics['r2']:.4f}"
    )

    comparison = load_baseline_comparison(split_info)
    if comparison.get("available") and comparison.get("same_test_period"):
        print("\nComparison on the identical test period (from results/model_metrics.json):")
        rows = [
            (MODEL_LABELS.get(k, k), v)
            for k, v in comparison["baseline_holdout_metrics"].items()
            if isinstance(v, dict) and "mae" in v
        ]
        rows.append((MODEL_LABELS[PYTORCH_MODEL_KEY], metrics))
        rows.sort(key=lambda kv: kv[1]["mae"])
        for name, m in rows:
            marker = " <- pytorch" if name == MODEL_LABELS[PYTORCH_MODEL_KEY] else ""
            print(
                f"  {name:<22} MAE={m['mae']:>12,.2f}  RMSE={m['rmse']:>12,.2f}"
                f"{marker}"
            )
    elif comparison.get("available"):
        print(
            "  WARNING: test periods differ between Phase 1 and Phase 3 — "
            "comparison suppressed."
        )
    else:
        print(f"  WARNING: {comparison.get('reason')}")

    # ------------------------------------------------------------------ save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
        },
        "model": PYTORCH_MODEL_KEY,
        "task": "panel_product_month",
        "seed": RANDOM_SEED,
        "features": FEATURE_COLUMNS,
        "split": split_info,
        "training": info,
        "metrics": metrics,
        "baseline_comparison": comparison,
        "notes": [
            "Same leakage-free features and target as Phase 1 "
            "(src.features.add_lag_features); no future rows are used.",
            "Same chronological holdout as Phase 1 "
            "(src.split.chronological_split with identical fraction).",
            "Standardization statistics fit on the training partition only; "
            "early stopping monitored on a tail of the training period, "
            "never on the test set.",
            "Baseline metrics under 'baseline_comparison' are loaded from "
            "results/model_metrics.json, not hardcoded.",
        ],
    }
    with open(TORCH_METRICS_PATH, "w") as f:
        json.dump(_round(results), f, indent=2)
    print(f"\nWrote {TORCH_METRICS_PATH}")

    predictions = pd.DataFrame(
        {
            "task": "panel_product_month",
            "period": test[MONTH_COLUMN].dt.strftime("%Y-%m"),
            "product": test[PRODUCT_COLUMN],
            "y_true": y_test.to_numpy(),
            "model": PYTORCH_MODEL_KEY,
            "y_pred": preds,
        }
    )
    predictions.to_csv(TORCH_PREDICTIONS_PATH, index=False)
    print(f"Wrote {TORCH_PREDICTIONS_PATH} ({len(predictions)} rows)")

    model.save(TORCH_MODEL_PATH)
    print(f"Wrote {TORCH_MODEL_PATH}")
    return results


def _round(obj, ndigits: int = 6):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, ndigits) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), ndigits)
    return obj


if __name__ == "__main__":
    main()
