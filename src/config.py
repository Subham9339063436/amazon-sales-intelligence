"""Central configuration: paths, features, split and model settings.

All values here are settings, not results. Actual evaluation results are
produced by ``scripts/run_training.py`` and stored in
``results/model_metrics.json``.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "Amazon_foodcategory_sales.csv"
SEGMENTS_PATH = DATA_DIR / "customer_segments.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
# Long-format holdout predictions written by scripts/run_training.py.
PREDICTIONS_PATH = RESULTS_DIR / "holdout_predictions.csv"
# Phase-3 PyTorch artifacts written by scripts/run_torch_training.py.
TORCH_METRICS_PATH = RESULTS_DIR / "pytorch_metrics.json"
TORCH_PREDICTIONS_PATH = RESULTS_DIR / "pytorch_holdout_predictions.csv"
TORCH_MODEL_PATH = MODELS_DIR / "pytorch_model.pt"

# ---------------------------------------------------------------------------
# Reproducibility / split settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
# Fraction of *observed months* (sorted chronologically) held out for testing.
# The exact train/test month counts depend on the modeling frame (rows with
# NaN lag features are dropped first); run scripts/run_training.py to see the
# resolved split, which is recorded in results/model_metrics.json.
TEST_FRACTION = 0.2
# Walk-forward (TimeSeriesSplit) folds computed on the training months only.
CV_SPLITS = 3
# Tail of the training period carved off as an early-stopping validation set
# for the final LightGBM model, so the test set is never used for model
# selection.
LIGHTGBM_ES_MONTHS = 3

# ---------------------------------------------------------------------------
# Modeling frame
# ---------------------------------------------------------------------------
TARGET_COLUMN = "sales_amount"
MONTH_COLUMN = "month"
PRODUCT_COLUMN = "product"

# Same feature set as the original notebook, except ``sales_rolling_3`` is now
# built strictly from *previous* observations (see src.features).
FEATURE_COLUMNS = [
    "year",
    "month_num",
    "quarter",
    "avg_price",
    "list_price",
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_3",
    "sales_rolling_3",
]

# ---------------------------------------------------------------------------
# Model parameters (kept equivalent to the original notebook)
# ---------------------------------------------------------------------------
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": ["l1", "rmse"],
    "learning_rate": 0.1,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
# Fixed rounds inside cross-validation (no early stopping on the fold being
# scored). The final holdout model may early-stop on a tail of the training
# data only.
LIGHTGBM_ROUNDS_CV = 300
LIGHTGBM_ROUNDS_FINAL = 1000
LIGHTGBM_EARLY_STOPPING_ROUNDS = 50

PROPHET_PARAMS = {
    "daily_seasonality": False,
    "weekly_seasonality": False,
    "yearly_seasonality": True,
    "changepoint_prior_scale": 0.05,
}

# ---------------------------------------------------------------------------
# PyTorch model (Phase 3)
# ---------------------------------------------------------------------------
# Same split methodology as Phase 1: features/target come from
# src.features (leakage-free), evaluation uses src.split.chronological_split.
TORCH_HIDDEN = (64, 32)
TORCH_LEARNING_RATE = 1e-3
TORCH_WEIGHT_DECAY = 1e-5
TORCH_MAX_EPOCHS = 500
TORCH_PATIENCE = 40
# Early-stopping tail carved from the *training* period — the same months
# LightGBM uses, so both models share an identical fit/validation partition
# and the chronological holdout stays untouched.
TORCH_ES_MONTHS = LIGHTGBM_ES_MONTHS

# Human-readable model names shared by the training scripts and dashboard.
MODEL_LABELS = {
    "naive_last_value": "Naive (last value)",
    "linear_regression": "Linear Regression",
    "lightgbm": "LightGBM",
    "prophet": "Prophet",
    "pytorch_mlp": "PyTorch (MLP)",
}
