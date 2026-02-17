"""Lightweight hyperparameter search for the unified regression models."""
from __future__ import annotations

import argparse
import random
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from utils import (
    DATA_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMNS,
    compute_regression_metrics,
    ensure_directories,
    load_dataset,
    prepare_features,
    prepare_targets,
    save_json,
    sequential_split,
    set_global_seed,
    temporal_train_val_split,
)

BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"

PARAM_SPACES: Dict[str, Dict] = {
    "xgboost": {
        "n_estimators": [400, 600, 800],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5],
        "gamma": [0.0, 0.1],
    },
    "random_forest": {
        "n_estimators": [300, 500, 700],
        "max_depth": [16, 20, None],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2],
        "max_features": [None, "sqrt"],
        "bootstrap": [True, False],
    },
    "mlp_regressor": {
        "hidden_layer_sizes": [(256, 128, 64), (128, 64, 32), (256, 128)],
        "learning_rate_init": [5e-4, 1e-3],
        "alpha": [1e-4, 5e-4, 1e-3],
        "max_iter": [400, 650],
        "batch_size": [64, 128],
        "activation": ["relu"],
    },
}


def sample_configs(space: Dict, max_configs: int, rng: random.Random) -> List[Dict]:
    grid = list(ParameterGrid(space))
    rng.shuffle(grid)
    return grid[: max_configs or len(grid)]


def build_model(model_name: str, params: Dict, random_state: int):
    if model_name == "xgboost":
        estimator = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        return MultiOutputRegressor(estimator)
    if model_name == "random_forest":
        return RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
    if model_name == "mlp_regressor":
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    MLPRegressor(
                        **params,
                        early_stopping=True,
                        n_iter_no_change=15,
                        validation_fraction=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return pipeline
    raise ValueError(f"Unknown model name: {model_name}")


def evaluate_model(model, X_train, y_train, X_val, y_val) -> Tuple[float, Dict[str, float]]:
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    _, overall = compute_regression_metrics(y_val, preds, TARGET_COLUMNS)
    return overall["rmse"], overall


def tune_models(max_configs: int, random_state: int) -> Dict[str, Dict]:
    ensure_directories()
    set_global_seed(random_state)
    rng = random.Random(random_state)

    df = load_dataset(DATA_PATH)
    train_df, _ = sequential_split(df)
    inner_train_df, val_df = temporal_train_val_split(train_df, val_ratio=0.2)

    X_train = prepare_features(inner_train_df)
    y_train = prepare_targets(inner_train_df)
    X_val = prepare_features(val_df)
    y_val = prepare_targets(val_df)

    best_params: Dict[str, Dict] = {}

    for model_name, space in PARAM_SPACES.items():
        print(f"\nTuning {model_name} (searching up to {max_configs} configs)...")
        configs = sample_configs(space, max_configs, rng)
        records = []
        best_rmse = float("inf")
        best_config: Dict | None = None
        for idx, params in enumerate(configs, start=1):
            model = build_model(model_name, params, random_state)
            rmse, overall = evaluate_model(model, X_train, y_train, X_val, y_val)
            record = {"config_id": idx, "rmse": rmse, **overall, **params}
            records.append(record)
            if rmse < best_rmse:
                best_rmse = rmse
                best_config = params
            print(f"  Config {idx}/{len(configs)} — RMSE: {rmse:.6f}")
        if best_config is None:
            continue
        best_params[model_name] = best_config
        results_df = pd.DataFrame(records).sort_values("rmse")
        results_path = MODELS_DIR / f"hparam_results_{model_name}.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Best {model_name} RMSE: {best_rmse:.6f} | Saved results to {results_path}")

    save_json(best_params, BEST_PARAMS_PATH)
    print(f"Saved best parameters to {BEST_PARAMS_PATH}")
    return best_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for unified models")
    parser.add_argument("--max-configs", type=int, default=8, help="Maximum configs to try per model")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tune_models(args.max_configs, args.seed)


if __name__ == "__main__":
    main()
