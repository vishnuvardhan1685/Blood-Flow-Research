"""Utility helpers for the unified fluid mechanics regression pipeline."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Unified_fluid_ml_dataset_60k.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

FEATURE_COLUMNS: List[str] = [
    "r",
    "tau_y",
    "Re",
    "Fr",
    "F",
    "k",
    "theta2",
    "alpha1",
    "alpha2",
    "eps1",
    "eps2",
]

TARGET_COLUMNS: List[str] = ["u", "uplv", "up", "tau_w"]

RANDOM_STATE = 42


def ensure_directories() -> None:
    """Create the folders used by the training scripts and dashboards."""
    for directory in (MODELS_DIR, FIGURES_DIR, DATA_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Set seeds for numpy and python's RNG to keep runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the unified fluid mechanics dataset."""
    csv_path = csv_path or DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def sequential_split(
    df: pd.DataFrame,
    n_parts: int = 5,
    train_parts: Sequence[int] = (0, 2, 4),
    test_parts: Sequence[int] = (1, 3),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sequentially split the frame into parts, keeping order intact."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    all_parts = set(train_parts) | set(test_parts)
    if not all(0 <= part < n_parts for part in all_parts):
        raise ValueError("Split indices must be within the number of parts")
    splits = np.array_split(df, n_parts)
    train_df = pd.concat([splits[i] for i in train_parts], axis=0).reset_index(drop=True)
    test_df = pd.concat([splits[i] for i in test_parts], axis=0).reset_index(drop=True)
    return train_df, test_df


def temporal_train_val_split(
    df: pd.DataFrame, val_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/validation chunks while preserving order."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    split_idx = int(len(df) * (1 - val_ratio))
    split_idx = max(1, min(len(df) - 1, split_idx))
    train_subset = df.iloc[:split_idx].reset_index(drop=True)
    val_subset = df.iloc[split_idx:].reset_index(drop=True)
    return train_subset, val_subset


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy()


def prepare_targets(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET_COLUMNS].to_numpy()


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str] = TARGET_COLUMNS,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return per-target metrics and overall summary."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match")
    metrics: List[Dict[str, float]] = []
    for idx, target in enumerate(target_names):
        y_true_col = y_true[:, idx]
        y_pred_col = y_pred[:, idx]
        mse = mean_squared_error(y_true_col, y_pred_col)
        metrics.append(
            {
                "target": target,
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_true_col, y_pred_col)),
                "mse": float(mse),
                "r2": float(r2_score(y_true_col, y_pred_col)),
            }
        )
    overall = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    metrics_df = pd.DataFrame(metrics)
    return metrics_df, overall


def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match for residuals")
    residuals = y_true - y_pred
    return pd.DataFrame(residuals, columns=target_names)


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(obj, path)


def load_joblib(path: Path):
    return load(path)


def plot_rmse_bar(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create a comparison bar chart for model RMSE."""
    if results_df.empty:
        return
    sorted_df = results_df.sort_values("overall_rmse")
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sorted_df["model"], sorted_df["overall_rmse"], color="#4c72b0")
    ax.set_ylabel("Overall RMSE")
    ax.set_title("Model RMSE Comparison")
    for idx, value in enumerate(sorted_df["overall_rmse"]):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    cols = len(target_names)
    n_rows = int(np.ceil(cols / 2))
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))
    axes = np.array(axes).flatten()
    for idx, target in enumerate(target_names):
        ax = axes[idx]
        ax.scatter(y_true[:, idx], y_pred[:, idx], alpha=0.4, edgecolor="none")
        ax.plot([y_true[:, idx].min(), y_true[:, idx].max()], [y_true[:, idx].min(), y_true[:, idx].max()], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(target)
    for ax in axes[cols:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    plt.style.use("seaborn-v0_8")
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Feature/Target Correlation Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residual_histograms(residuals: pd.DataFrame, output_path: Path) -> None:
    if residuals.empty:
        return
    cols = len(residuals.columns)
    n_rows = int(np.ceil(cols / 2))
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5 * n_rows))
    axes = np.array(axes).flatten()
    for idx, column in enumerate(residuals.columns):
        ax = axes[idx]
        sns.histplot(residuals[column], bins=40, kde=True, ax=ax, color="#dd8452")
        ax.set_title(f"Residuals — {column}")
        ax.set_xlabel("Residual")
    for ax in axes[cols:]:
        ax.axis("off")
    fig.suptitle("Residual Distributions by Target")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_residual_scatter(
    y_true: np.ndarray,
    residuals: pd.DataFrame,
    target_names: Sequence[str],
    output_path: Path,
) -> None:
    if residuals.empty:
        return
    cols = len(target_names)
    n_rows = int(np.ceil(cols / 2))
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5 * n_rows))
    axes = np.array(axes).flatten()
    for idx, target in enumerate(target_names):
        ax = axes[idx]
        ax.scatter(y_true[:, idx], residuals[target], alpha=0.35, color="#55a868", edgecolor="none")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Residual")
        ax.set_title(f"Residual vs Actual — {target}")
    for ax in axes[cols:]:
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def extract_feature_importances(model, feature_names: Sequence[str]) -> pd.Series | None:
    """Return averaged feature importances for tree ensembles when available."""
    estimator = model
    if hasattr(estimator, "named_steps"):
        estimator = estimator.named_steps.get("regressor", estimator.named_steps.get("model", estimator))
    if isinstance(estimator, MultiOutputRegressor):
        importances = []
        for est in estimator.estimators_:
            if hasattr(est, "feature_importances_"):
                importances.append(est.feature_importances_)
        if not importances:
            return None
        mean_importance = np.mean(importances, axis=0)
    elif hasattr(estimator, "feature_importances_"):
        mean_importance = estimator.feature_importances_
    else:
        return None
    return pd.Series(mean_importance, index=feature_names).sort_values(ascending=False)


def plot_feature_importance(series: pd.Series, output_path: Path, title: str) -> None:
    if series is None or series.empty:
        return
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 5))
    series.iloc[::-1].plot(kind="barh", ax=ax, color="#55a868")
    ax.set_xlabel("Mean importance")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


@dataclass
class ModelRecord:
    name: str
    estimator: object
    per_target_metrics: pd.DataFrame
    overall_metrics: Dict[str, float]

    def to_summary_dict(self) -> Dict[str, float]:
        summary = {"model": self.name}
        summary.update({f"overall_{key}": value for key, value in self.overall_metrics.items()})
        return summary
