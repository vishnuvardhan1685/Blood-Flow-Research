"""Training script for the unified multi-output regression models."""
from __future__ import annotations

import argparse
from typing import Dict

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from utils import (
    DATA_PATH,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_COLUMNS,
    ModelRecord,
    compute_residuals,
    compute_regression_metrics,
    ensure_directories,
    extract_feature_importances,
    load_dataset,
    load_json,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_prediction_scatter,
    plot_rmse_bar,
    plot_residual_histograms,
    plot_residual_scatter,
    prepare_features,
    prepare_targets,
    save_json,
    sequential_split,
    set_global_seed,
)

BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"

DEFAULT_XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.0,
    "reg_lambda": 1.0,
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 18,
    "min_samples_split": 4,
    "min_samples_leaf": 1,
    "max_features": None,
}

DEFAULT_MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "relu",
    "learning_rate_init": 1e-3,
    "max_iter": 500,
    "early_stopping": True,
    "n_iter_no_change": 20,
    "validation_fraction": 0.1,
    "alpha": 1e-4,
    "batch_size": "auto",
}


def build_models(random_state: int, overrides: Dict[str, Dict] | None = None) -> Dict[str, object]:
    """Instantiate the three required regressors."""
    overrides = overrides or {}

    xgb_params = {**DEFAULT_XGB_PARAMS, **overrides.get("xgboost", {})}
    xgb = MultiOutputRegressor(
        XGBRegressor(
            **xgb_params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    )

    rf_params = {**DEFAULT_RF_PARAMS, **overrides.get("random_forest", {})}
    rf = RandomForestRegressor(**rf_params, random_state=random_state, n_jobs=-1)

    mlp_params = {**DEFAULT_MLP_PARAMS, **overrides.get("mlp_regressor", {})}
    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    **mlp_params,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "xgboost": xgb,
        "random_forest": rf,
        "mlp_regressor": mlp,
    }


def train_models(random_state: int) -> Dict[str, ModelRecord]:
    ensure_directories()
    set_global_seed(random_state)
    df = load_dataset(DATA_PATH)
    train_df, test_df = sequential_split(df)

    X_train = prepare_features(train_df)
    y_train = prepare_targets(train_df)
    X_test = prepare_features(test_df)
    y_test = prepare_targets(test_df)

    param_overrides = load_json(BEST_PARAMS_PATH) if BEST_PARAMS_PATH.exists() else {}
    if param_overrides:
        print("Loaded tuned hyperparameters from best_params.json")

    models = build_models(random_state, param_overrides)
    records: Dict[str, ModelRecord] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        per_target_df, overall_metrics = compute_regression_metrics(y_test, y_pred, TARGET_COLUMNS)
        records[name] = ModelRecord(name, model, per_target_df, overall_metrics)
        print(f"\n{name.upper()} PERFORMANCE")
        print(per_target_df)
        print("Overall:", overall_metrics)

    summary_rows = [record.to_summary_dict() for record in records.values()]
    summary_df = pd.DataFrame(summary_rows).sort_values("overall_rmse")
    summary_path = MODELS_DIR / "model_results.csv"
    summary_df.to_csv(summary_path, index=False)
    plot_rmse_bar(summary_df, FIGURES_DIR / "model_rmse.png")

    target_metrics_df = pd.concat(
        [rec.per_target_metrics.assign(model=rec.name) for rec in records.values()], axis=0
    )
    target_metrics_df.to_csv(MODELS_DIR / "per_target_metrics.csv", index=False)

    best_name = summary_df.iloc[0]["model"]
    best_record = records[best_name]
    best_model_path = MODELS_DIR / "best_model.pkl"
    dump(best_record.estimator, best_model_path)

    best_preds = best_record.estimator.predict(X_test)
    plot_prediction_scatter(
        y_test,
        best_preds,
        TARGET_COLUMNS,
        f"Predicted vs Actual ({best_name})",
        FIGURES_DIR / f"{best_name}_pred_vs_actual.png",
    )

    residuals_df = compute_residuals(y_test, best_preds, TARGET_COLUMNS)
    plot_residual_histograms(residuals_df, FIGURES_DIR / f"{best_name}_residual_hist.png")
    plot_residual_scatter(
        y_test,
        residuals_df,
        TARGET_COLUMNS,
        FIGURES_DIR / f"{best_name}_residual_scatter.png",
    )
    plot_correlation_heatmap(
        df[FEATURE_COLUMNS + TARGET_COLUMNS], FIGURES_DIR / "feature_target_correlation.png"
    )

    for model_name, record in records.items():
        fi_series = extract_feature_importances(record.estimator, FEATURE_COLUMNS)
        if fi_series is not None:
            plot_feature_importance(
                fi_series,
                FIGURES_DIR / f"{model_name}_feature_importance.png",
                f"Feature Importance — {model_name}",
            )

    feature_stats = {
        col: {
            "min": float(train_df[col].min()),
            "max": float(train_df[col].max()),
            "mean": float(train_df[col].mean()),
        }
        for col in FEATURE_COLUMNS
    }

    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "best_model_overall_metrics": best_record.overall_metrics,
        "best_model_per_target_metrics": best_record.per_target_metrics.to_dict(orient="records"),
        "feature_stats": feature_stats,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "results_table": str(summary_path.relative_to(PROJECT_ROOT)),
        "best_model_hyperparameters": param_overrides.get(best_name, {})
        if param_overrides
        else {},
        "research_reference": "Insights from the attached KL-Newtonian fluid study informed the focus on Froude, Reynolds, permeability, and yield-stress parameters when tuning these regressors.",
    }
    save_json(metadata, MODELS_DIR / "model_metadata.json")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare multi-output regressors")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_models(args.seed)


if __name__ == "__main__":
    main()
