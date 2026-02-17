"""Streamlit dashboard for the unified multi-output regression models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    TARGET_COLUMNS,
    load_joblib,
    load_json,
)

RESULTS_PATH = MODELS_DIR / "model_results.csv"
PER_TARGET_PATH = MODELS_DIR / "per_target_metrics.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def load_artifacts() -> Tuple[object, Dict, pd.DataFrame, pd.DataFrame]:
    if not (RESULTS_PATH.exists() and PER_TARGET_PATH.exists() and BEST_MODEL_PATH.exists() and METADATA_PATH.exists()):
        raise FileNotFoundError(
            "Required artifacts are missing. Please run train.py first to generate the trained models and metadata."
        )
    model = load_joblib(BEST_MODEL_PATH)
    metadata = load_json(METADATA_PATH)
    summary_df = pd.read_csv(RESULTS_PATH)
    per_target_df = pd.read_csv(PER_TARGET_PATH)
    return model, metadata, summary_df, per_target_df


def render_sidebar(metadata: Dict) -> None:
    st.sidebar.header("Best Model Snapshot")
    st.sidebar.write(f"**Selected Model:** {metadata['best_model_name'].replace('_', ' ').title()}")
    overall = metadata.get("best_model_overall_metrics", {})
    st.sidebar.metric("Overall RMSE", f"{overall.get('rmse', float('nan')):.5f}")
    st.sidebar.metric("Overall MAE", f"{overall.get('mae', float('nan')):.5f}")
    st.sidebar.metric("Overall R²", f"{overall.get('r2', float('nan')):.4f}")
    best_hparams = metadata.get("best_model_hyperparameters") or {}
    if best_hparams:
        st.sidebar.subheader("Tuned Hyperparameters")
        st.sidebar.json(best_hparams)
    st.sidebar.caption(
        "Model tuning emphasized KL-Newtonian controls (Fr, Re, τy, θ₂, permeability) highlighted in the provided research article."
    )


def build_input_form(metadata: Dict) -> pd.DataFrame:
    st.subheader("Input Fluid-Mechanics Parameters")
    feature_stats = metadata.get("feature_stats", {})
    cols = st.columns(2)
    values = {}
    for idx, feature in enumerate(FEATURE_COLUMNS):
        stats = feature_stats.get(feature, {})
        min_val = stats.get("min", 0.0)
        max_val = stats.get("max", min_val + 1.0)
        if np.isclose(max_val, min_val):
            max_val = min_val + 1.0
        default = stats.get("mean", (min_val + max_val) / 2)
        step = (max_val - min_val) / 100 if max_val > min_val else 0.01
        with cols[idx % 2]:
            values[feature] = st.number_input(
                feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default),
                step=float(step),
                format="%.6f",
            )
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def main() -> None:
    st.set_page_config(page_title="Unified Fluid Regression Dashboard", layout="wide")
    st.title("Unified Multi-Output Regression for Fluid Mechanics")
    st.write(
        "Predictions are driven by the high-resolution KL-Newtonian dataset and follow the sequential split protocol to test interpolation across ordered regimes."
    )

    try:
        model, metadata, summary_df, per_target_df = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(metadata)
    user_df = build_input_form(metadata)

    if st.button("Predict Hemodynamic Outputs", type="primary"):
        preds = model.predict(user_df[FEATURE_COLUMNS])
        pred_df = pd.DataFrame(preds, columns=TARGET_COLUMNS)
        st.success("Predictions ready")
        st.dataframe(pred_df.style.format(precision=6), use_container_width=True)

    st.subheader("Model Comparison (Overall Metrics)")
    st.dataframe(summary_df.style.format(precision=6), use_container_width=True)

    st.subheader("Per-Target Metrics for the Selected Model")
    best_name = metadata.get("best_model_name")
    per_target_best = per_target_df[per_target_df["model"] == best_name]
    st.dataframe(per_target_best.style.format(precision=6), use_container_width=True)

    rmse_plot = Path("figures/model_rmse.png")
    scatter_plot = Path(f"figures/{best_name}_pred_vs_actual.png")
    residual_hist = Path(f"figures/{best_name}_residual_hist.png")
    residual_scatter = Path(f"figures/{best_name}_residual_scatter.png")
    corr_plot = Path("figures/feature_target_correlation.png")

    col1, col2 = st.columns(2)
    if rmse_plot.exists():
        col1.image(str(rmse_plot), caption="RMSE comparison across models", use_column_width=True)
    if scatter_plot.exists():
        col2.image(str(scatter_plot), caption="Predicted vs Actual for best model", use_column_width=True)

    col3, col4 = st.columns(2)
    if residual_hist.exists():
        col3.image(str(residual_hist), caption="Residual distributions per output", use_column_width=True)
    if residual_scatter.exists():
        col4.image(str(residual_scatter), caption="Residual vs Actual", use_column_width=True)

    if corr_plot.exists():
        st.image(str(corr_plot), caption="Feature/Target Correlation Matrix", use_column_width=True)

    st.markdown(
        "_Reference_: Kuntal et al. (2026) 'Asymptotic analysis of a multi-layered model of KL-Newtonian fluids' informs the emphasis on Froude, Reynolds, permeability, and yield-stress terms used throughout the pipeline."
    )


if __name__ == "__main__":
    main()
