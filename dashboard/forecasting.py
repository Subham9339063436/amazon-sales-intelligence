"""Forecasting tab: actual vs predicted + metrics loaded from results files.

Every number on this tab comes from ``results/model_metrics.json`` or
``results/holdout_predictions.csv`` produced by ``scripts/run_training.py``.
Nothing is hardcoded; if the files are missing the tab explains what to run.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data import (
    MODEL_LABELS,
    get_metrics,
    get_predictions,
    get_torch_metrics,
    get_torch_predictions,
    results_available,
)

METRIC_FORMAT = {
    "mae": ("MAE", "${:,.2f}"),
    "rmse": ("RMSE", "${:,.2f}"),
    "mape_percent": ("MAPE", "{:.2f}%"),
    "r2": ("R²", "{:.4f}"),
}


def _metrics_table(metric_blocks: dict[str, dict]) -> pd.DataFrame:
    """Flatten a {model: {mae, rmse, mape_percent, r2}} mapping to a table."""
    rows = []
    for model_key, block in metric_blocks.items():
        if not isinstance(block, dict) or "mae" not in block:
            continue  # skip nested extra fields (e.g. best_iteration)
        row = {"Model": MODEL_LABELS.get(model_key, model_key)}
        for key, (label, fmt) in METRIC_FORMAT.items():
            value = block.get(key)
            row[label] = fmt.format(value) if value is not None else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_series(metric_blocks: dict[str, dict], metric_key: str) -> pd.Series:
    """Raw numeric series of one metric per model (for bar charts)."""
    values = {}
    for model_key, block in metric_blocks.items():
        if isinstance(block, dict) and block.get(metric_key) is not None:
            values[MODEL_LABELS.get(model_key, model_key)] = block[metric_key]
    return pd.Series(values, dtype=float)


def _cv_table(cv_summary: dict) -> pd.DataFrame:
    rows = []
    for model_key, stats in cv_summary.items():
        row = {"Model": MODEL_LABELS.get(model_key, model_key)}
        for metric_key in ("mae", "rmse"):
            s = stats.get(metric_key)
            if isinstance(s, dict):
                row[f"{METRIC_FORMAT[metric_key][0]} mean"] = s["mean"]
                row[f"{METRIC_FORMAT[metric_key][0]} std"] = s["std"]
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def render() -> None:
    st.subheader("Forecasting")

    available = results_available()
    if not available["metrics"] or not available["predictions"]:
        st.warning(
            "Result files not found. Run the pipeline first:\n\n"
            "`python scripts/run_training.py`\n\n"
            "Expected files: `results/model_metrics.json`, "
            "`results/holdout_predictions.csv`."
        )
        return

    metrics = get_metrics()
    preds = get_predictions()

    # Phase-3 PyTorch results (optional file; merged when present).
    torch_bundle = None
    if available["torch_metrics"] and available["torch_predictions"]:
        torch_bundle = (get_torch_metrics(), get_torch_predictions())

    split = metrics["split"]
    st.caption(
        f"Evaluation generated {metrics['generated_at_utc']} · "
        f"train {split['train_month_range'][0]} → {split['train_month_range'][1]} "
        f"({split['n_train_months']} months), holdout "
        f"{split['test_month_range'][0]} → {split['test_month_range'][1]} "
        f"({split['n_test_months']} months). All metrics below are loaded from "
        "results/model_metrics.json — not hardcoded."
    )

    # ------------------------------------------------------------------
    # Model comparison tables (holdout metrics from the results file)
    # ------------------------------------------------------------------
    st.markdown("##### Model comparison — holdout metrics")
    t1, t2 = st.columns(2)

    panel_blocks = dict(metrics["holdout_panel_product_month"]["metrics"])
    revenue_blocks = metrics["holdout_revenue_monthly_total"]["metrics"]
    if torch_bundle is not None:
        torch_m = torch_bundle[0]["metrics"]
        panel_blocks["pytorch_mlp"] = {
            k: torch_m[k] for k in ("mae", "rmse", "mape_percent", "r2")
        }

    with t1:
        st.caption("Task A: product-month sales (panel)")
        table_a = _metrics_table(panel_blocks)
        # Best model by holdout MAE — computed from the loaded metric files.
        mae_by_label = {
            MODEL_LABELS.get(k, k): v["mae"]
            for k, v in panel_blocks.items()
            if isinstance(v, dict) and v.get("mae") is not None
        }
        if mae_by_label:
            best_label = min(mae_by_label, key=mae_by_label.get)
            st.caption(f"Lowest holdout MAE: **{best_label}** (from loaded metrics)")
        st.dataframe(table_a, use_container_width=True, hide_index=True)

    with t2:
        st.caption("Task B: total monthly revenue")
        st.dataframe(
            _metrics_table(revenue_blocks), use_container_width=True, hide_index=True
        )

    # --- Walk-forward CV summary ----------------------------------------
    with st.expander("Walk-forward CV summary (training months only)"):
        st.dataframe(
            _cv_table(metrics["walk_forward_cv"]["summary"]),
            use_container_width=True,
            hide_index=True,
        )

    # --- Metric bar charts -----------------------------------------------
    st.markdown("##### Metric comparison")
    metric_key = st.selectbox(
        "Metric",
        options=list(METRIC_FORMAT.keys()),
        format_func=lambda k: METRIC_FORMAT[k][0],
        key="fc_metric",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Product-month sales task")
        series_a = _metric_series(panel_blocks, metric_key)
        if not series_a.empty:
            st.bar_chart(series_a, height=300)
    with c2:
        st.caption("Total monthly revenue task")
        series_b = _metric_series(revenue_blocks, metric_key)
        if not series_b.empty:
            st.bar_chart(series_b, height=300)

    # ------------------------------------------------------------------
    # Actual vs predicted (from holdout_predictions.csv)
    # ------------------------------------------------------------------
    st.markdown("##### Actual vs predicted (holdout)")

    revenue_preds = preds[preds["task"] == "revenue_monthly_total"]
    panel_preds = preds[preds["task"] == "panel_product_month"]

    if not revenue_preds.empty:
        st.caption("Total monthly revenue — actual vs naive vs Prophet")
        rev_wide = revenue_preds.pivot_table(
            index="period", columns="model", values="y_pred", aggfunc="first"
        )
        actual_by_period = (
            revenue_preds.drop_duplicates("period")
            .set_index("period")["y_true"]
            .astype(float)
        )
        plot_df = pd.DataFrame({"Actual": actual_by_period})
        for model_key in rev_wide.columns:
            plot_df[MODEL_LABELS.get(model_key, model_key)] = rev_wide[model_key]
        st.line_chart(plot_df, height=350)

    if torch_bundle is not None:
        panel_preds = pd.concat(
            [panel_preds, torch_bundle[1]], ignore_index=True, sort=False
        )

    if not panel_preds.empty:
        st.caption("Product-month sales — actual vs predicted (5 test months)")
        panel_models = sorted(panel_preds["model"].unique())
        panel_sel = st.multiselect(
            "Models",
            options=panel_models,
            default=panel_models,
            format_func=lambda m: MODEL_LABELS.get(m, m),
            key="fc_panel_models",
        )
        if panel_sel:
            sample = panel_preds[panel_preds["model"].isin(panel_sel)].copy()
            # Per-model, per-month MAE is the most readable comparison for
            # 1,500+ scattered points; a sample table is shown underneath.
            sample["abs_error"] = (sample["y_true"] - sample["y_pred"]).abs()
            mae_by_model_month = (
                sample.groupby(["model", "period"])["abs_error"]
                .mean()
                .reset_index()
            )
            mae_wide = mae_by_model_month.pivot(
                index="period", columns="model", values="abs_error"
            )
            mae_wide.columns = [MODEL_LABELS.get(m, m) for m in mae_wide.columns]
            st.caption("Mean absolute error by holdout month (lower is better)")
            st.line_chart(mae_wide, height=300)

            st.markdown("Sample of holdout rows (first 20, best model LightGBM):")
            lgbm_rows = panel_preds[panel_preds["model"] == "lightgbm"].head(20).copy()
            lgbm_rows["error"] = (lgbm_rows["y_true"] - lgbm_rows["y_pred"]).round(2)
            show = lgbm_rows[
                ["period", "product", "y_true", "y_pred", "error"]
            ].rename(
                columns={
                    "period": "Month",
                    "product": "Product",
                    "y_true": "Actual ($)",
                    "y_pred": "Predicted ($)",
                    "error": "Error ($)",
                }
            )
            st.dataframe(show, use_container_width=True, hide_index=True)

    # --- Prophet forecast table (from the results JSON) -------------------
    prophet_rows = metrics["holdout_revenue_monthly_total"].get("prophet_forecast", [])
    if prophet_rows:
        with st.expander("Prophet holdout forecast vs actual (from results JSON)"):
            st.dataframe(
                pd.DataFrame(prophet_rows).rename(
                    columns={
                        "ds": "Month",
                        "actual": "Actual ($)",
                        "predicted": "Predicted ($)",
                        "yhat_lower": "Lower 80% ($)",
                        "yhat_upper": "Upper 80% ($)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # --- Phase-3 comparison note (loaded, not asserted) -------------------
    if torch_bundle is not None:
        torch_result = torch_bundle[0]
        comp = torch_result.get("baseline_comparison", {})
        status = (
            "same test period verified"
            if comp.get("same_test_period")
            else "test period mismatch — comparison unreliable"
        )
        st.caption(
            f"PyTorch (MLP) loaded from `results/pytorch_metrics.json` — "
            f"seed {torch_result.get('seed')}, "
            f"{torch_result['training']['epochs_run']} epochs "
            f"(best {torch_result['training']['best_epoch']}); {status}."
        )
    else:
        st.caption(
            "PyTorch results not found — run `python scripts/run_torch_training.py` "
            "to add them to this comparison."
        )

    # --- Evaluation notes from the results file ---------------------------
    notes = list(metrics.get("notes", []))
    if torch_bundle is not None:
        notes += torch_bundle[0].get("notes", [])
    if notes:
        with st.expander("Evaluation notes (from results/model_metrics.json)"):
            for note in notes:
                st.markdown(f"- {note}")
