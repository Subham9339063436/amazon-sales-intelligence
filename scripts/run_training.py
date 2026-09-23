"""End-to-end training and evaluation for the forecasting pipeline.

Runs, in order:
  1. Load raw data, build the monthly product panel and revenue series.
  2. Add strictly past-only lag/rolling features.
  3. Chronological train/test split (train months always precede test months).
  4. Walk-forward (TimeSeriesSplit on months) validation on training data for
     Linear Regression, LightGBM, and a naive persistence baseline.
  5. Final holdout evaluation of naive / Linear Regression / LightGBM on the
     product panel, and of naive / Prophet on total monthly revenue.
  6. Write machine-readable results to ``results/model_metrics.json``.

Every metric written to disk is computed here from real predictions on data
the model did not train on. Nothing is hard-coded.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as ``python scripts/run_training.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import NaiveLastValueModel, last_observed_value
from src.config import (
    FEATURE_COLUMNS,
    LIGHTGBM_ES_MONTHS,
    LIGHTGBM_EARLY_STOPPING_ROUNDS,
    LIGHTGBM_ROUNDS_CV,
    METRICS_PATH,
    MONTH_COLUMN,
    PRODUCT_COLUMN,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_FRACTION,
    CV_SPLITS,
)
from src.data import build_monthly_panel, build_total_monthly_revenue, load_raw
from src.features import add_lag_features
from src.forecasting import fit_prophet, forecast_holdout
from src.metrics import regression_metrics
from src.models import fit_predict_lightgbm, fit_predict_linear
from src.split import chronological_split, walk_forward_folds

np.random.seed(RANDOM_SEED)


def _round(obj, ndigits: int = 6):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, ndigits) for v in obj]
    return obj


def _month_span(months: pd.Series | np.ndarray) -> list[str]:
    months = pd.Series(months).sort_values()
    return [months.iloc[0].strftime("%Y-%m"), months.iloc[-1].strftime("%Y-%m")]


def run_walk_forward_cv(modeling: pd.DataFrame) -> dict:
    """Walk-forward CV over training months for the panel models."""
    folds_out = []
    for fold in walk_forward_folds(modeling, n_splits=CV_SPLITS):
        tr, va = fold["train"], fold["val"]
        X_tr = tr[FEATURE_COLUMNS]
        y_tr = tr[TARGET_COLUMN]
        X_va = va[FEATURE_COLUMNS]
        y_va = va[TARGET_COLUMN]

        naive = NaiveLastValueModel().fit(tr)
        naive_pred = naive.predict(va)

        lr_pred = fit_predict_linear(X_tr, y_tr, X_va)
        lgbm_pred, _ = fit_predict_lightgbm(
            X_tr,
            y_tr,
            X_va,
            num_boost_round=LIGHTGBM_ROUNDS_CV,
        )

        folds_out.append(
            {
                "fold": fold["fold"],
                "train_range": fold["train_range"],
                "val_range": fold["val_range"],
                "n_train_rows": int(len(tr)),
                "n_val_rows": int(len(va)),
                "metrics": {
                    "naive_last_value": regression_metrics(y_va, naive_pred),
                    "linear_regression": regression_metrics(y_va, lr_pred),
                    "lightgbm": regression_metrics(y_va, lgbm_pred),
                },
            }
        )

    summary = {}
    for model_name in ["naive_last_value", "linear_regression", "lightgbm"]:
        summary[model_name] = {}
        for metric in ["mae", "rmse", "mape_percent", "r2"]:
            values = [
                f["metrics"][model_name][metric]
                for f in folds_out
                if f["metrics"][model_name][metric] is not None
            ]
            if values:
                summary[model_name][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n_folds": len(values),
                }
            else:
                summary[model_name][metric] = None
    return {"n_splits": CV_SPLITS, "folds": folds_out, "summary": summary}


def run_holdout_panel(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Holdout evaluation of naive / LR / LightGBM on product-month rows.

    Returns the metrics dict plus a long-format predictions frame
    (one row per test observation per model) for downstream dashboards.
    """
    X_tr = train[FEATURE_COLUMNS]
    y_tr = train[TARGET_COLUMN]
    X_te = test[FEATURE_COLUMNS]
    y_te = test[TARGET_COLUMN]

    naive = NaiveLastValueModel().fit(train)
    naive_pred = naive.predict(test)

    lr_pred = fit_predict_linear(X_tr, y_tr, X_te)

    # Early-stopping tail: last LIGHTGBM_ES_MONTHS of the *training* period.
    train_months = np.sort(train[MONTH_COLUMN].unique())
    n_es = min(LIGHTGBM_ES_MONTHS, max(1, len(train_months) // 4))
    es_months = train_months[-n_es:]
    fit_mask = ~train[MONTH_COLUMN].isin(es_months)
    es_frame = train[train[MONTH_COLUMN].isin(es_months)]

    lgbm_pred, lgbm_model = fit_predict_lightgbm(
        X_tr[fit_mask],
        y_tr[fit_mask],
        X_te,
        X_early_stop=es_frame[FEATURE_COLUMNS],
        y_early_stop=es_frame[TARGET_COLUMN],
        num_boost_round=1000,
        early_stopping_rounds=LIGHTGBM_EARLY_STOPPING_ROUNDS,
    )

    results = {
        "test_month_span": _month_span(test[MONTH_COLUMN].unique()),
        "metrics": {
            "naive_last_value": {
                **regression_metrics(y_te, naive_pred),
                "n_fallback_products": naive.n_fallback_,
            },
            "linear_regression": regression_metrics(y_te, lr_pred),
            "lightgbm": {
                **regression_metrics(y_te, lgbm_pred),
                "best_iteration": int(lgbm_model.best_iteration or 0),
                "early_stop_month_span": _month_span(es_months),
            },
        },
    }

    base = pd.DataFrame(
        {
            "period": test[MONTH_COLUMN].dt.strftime("%Y-%m"),
            "product": test[PRODUCT_COLUMN],
            "y_true": y_te.to_numpy(),
        }
    )
    predictions = pd.concat(
        [
            base.assign(model="naive_last_value", y_pred=naive_pred),
            base.assign(model="linear_regression", y_pred=lr_pred),
            base.assign(model="lightgbm", y_pred=lgbm_pred),
        ],
        ignore_index=True,
    )
    predictions.insert(0, "task", "panel_product_month")
    return results, predictions


def run_holdout_revenue(
    train_monthly: pd.DataFrame,
    test_monthly: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Holdout evaluation of naive / Prophet on total monthly revenue.

    Returns the metrics dict plus long-format predictions for the dashboard.
    """
    baseline_value = last_observed_value(train_monthly)
    naive_pred = np.full(len(test_monthly), baseline_value)

    prophet_model = fit_prophet(train_monthly)
    prophet_frame, prophet_metrics = forecast_holdout(
        prophet_model, train_monthly, test_monthly
    )

    results = {
        "test_month_span": [
            test_monthly["ds"].min().strftime("%Y-%m"),
            test_monthly["ds"].max().strftime("%Y-%m"),
        ],
        "metrics": {
            "naive_last_value": regression_metrics(
                test_monthly["y"].to_numpy(), naive_pred
            ),
            "prophet": prophet_metrics,
        },
        "prophet_forecast": [
            {
                "ds": row.ds.strftime("%Y-%m"),
                "actual": round(float(row.y), 2),
                "predicted": round(float(row.yhat), 2),
                "yhat_lower": round(float(row.yhat_lower), 2),
                "yhat_upper": round(float(row.yhat_upper), 2),
            }
            for row in prophet_frame.itertuples(index=False)
        ],
    }

    # prophet_frame is the test frame merged with forecasts (same row order).
    rev_base = pd.DataFrame(
        {
            "period": prophet_frame["ds"].dt.strftime("%Y-%m"),
            "product": "__ALL_PRODUCTS__",
            "y_true": prophet_frame["y"].to_numpy(),
        }
    )
    predictions = pd.concat(
        [
            rev_base.assign(model="naive_last_value", y_pred=naive_pred),
            rev_base.assign(model="prophet", y_pred=prophet_frame["yhat"].to_numpy()),
        ],
        ignore_index=True,
    )
    predictions.insert(0, "task", "revenue_monthly_total")
    return results, predictions


def main() -> dict:
    print("Loading data ...")
    raw = load_raw()
    panel = build_monthly_panel(raw)
    modeling = add_lag_features(panel)
    modeling = modeling.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    revenue = build_total_monthly_revenue(raw)

    print(f"  raw rows:            {len(raw)}")
    print(f"  product-month rows:  {len(panel)}")
    print(f"  modeling rows:       {len(modeling)} (after dropping NaN lags)")
    print(f"  observed months:     {panel[MONTH_COLUMN].nunique()}")

    train, test, split_info = chronological_split(modeling, TEST_FRACTION)
    print(
        f"  split: train {split_info['train_month_range'][0]}"
        f"..{split_info['train_month_range'][1]} ({split_info['n_train_rows']} rows), "
        f"test {split_info['test_month_range'][0]}"
        f"..{split_info['test_month_range'][1]} ({split_info['n_test_rows']} rows)"
    )

    # Revenue frames use the same chronological boundary as the panel split.
    cutoff = pd.Timestamp(split_info["train_month_range"][1]) + pd.offsets.MonthEnd(1)
    train_rev = revenue[revenue["ds"] <= cutoff]
    test_rev = revenue[revenue["ds"] > cutoff]
    assert train_rev["ds"].max() < test_rev["ds"].min()

    print("\nWalk-forward validation on training months ...")
    cv_results = run_walk_forward_cv(train)
    for name, s in cv_results["summary"].items():
        mae, rmse = s["mae"], s["rmse"]
        print(
            f"  {name:<22} CV MAE={mae['mean']:>12,.2f} ±{mae['std']:>10,.2f}  "
            f"RMSE={rmse['mean']:>12,.2f}"
        )

    print("\nHoldout evaluation (panel: product-month sales) ...")
    panel_results, panel_preds = run_holdout_panel(train, test)
    for name, m in panel_results["metrics"].items():
        print(
            f"  {name:<22} MAE={m['mae']:>12,.2f}  RMSE={m['rmse']:>12,.2f}  "
            f"MAPE={m['mape_percent']:.2f}%  R2={m['r2']:.4f}"
        )

    print("\nHoldout evaluation (total monthly revenue) ...")
    revenue_results, revenue_preds = run_holdout_revenue(train_rev, test_rev)
    for name, m in revenue_results["metrics"].items():
        print(
            f"  {name:<22} MAE={m['mae']:>12,.2f}  RMSE={m['rmse']:>12,.2f}  "
            f"MAPE={m['mape_percent']:.2f}%  R2={m['r2']:.4f}"
        )

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data": {
            "raw_rows": int(len(raw)),
            "product_month_rows": int(len(panel)),
            "modeling_rows": int(len(modeling)),
            "observed_months": int(panel[MONTH_COLUMN].nunique()),
            "month_span": _month_span(panel[MONTH_COLUMN].unique()),
        },
        "split": split_info,
        "features": FEATURE_COLUMNS,
        "walk_forward_cv": cv_results,
        "holdout_panel_product_month": panel_results,
        "holdout_revenue_monthly_total": revenue_results,
        "notes": [
            "All metrics computed on holdout/fold data not used for fitting "
            "(walk-forward folds for CV; held-out final months as untouched "
            "holdout).",
            "sales_rolling_3 is the mean of the three *previous* observations "
            "(t-3..t-1) and excludes the current row; the original notebook's "
            "rolling feature included the current row (target leakage).",
            "LightGBM early stopping used only the tail of the training "
            "period; the test set was never used for model selection.",
            "MAPE reported because the target (sales amount) is strictly "
            "positive in this dataset.",
            "Data contains a gap: 2018-04..2018-12 have no observations; "
            "lags therefore reference previous *observed* months.",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(_round(results), f, indent=2)
    print(f"\nWrote {METRICS_PATH}")

    # Long-format holdout predictions for the dashboard (actual vs predicted).
    predictions = pd.concat([panel_preds, revenue_preds], ignore_index=True)
    pred_path = RESULTS_DIR / "holdout_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"Wrote {pred_path} ({len(predictions)} rows)")
    return results


if __name__ == "__main__":
    main()
