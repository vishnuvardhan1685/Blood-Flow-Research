# Unified Multi-Output Regression for Fluid Mechanics

This repository trains a unified multi-output regression system that predicts the four key hemodynamic targets $(u, u_{plv}, u_p, \tau_w)$ from eleven governing parameters drawn from the **60k-row unified fluid mechanics dataset**. The workflow mirrors the sequential, layered insights presented in Kuntal et al. (2026), who analysed KL-Newtonian blood-flow dynamics with spatial variations in permeability, viscosity, and Froude/Reynolds controls. Those findings motivated our focus on Froude number, Reynolds number, yield stress, Kuang–Luo parameters $(\theta_2)$, and porous permeability pairs $(\alpha, \varepsilon)$ during model design and evaluation.

## Project structure

```
├── data/
│   └── Unified_fluid_ml_dataset_60k.csv
├── figures/
│   ├── feature_target_correlation.png
│   ├── model_rmse.png
│   ├── xgboost_pred_vs_actual.png
│   ├── xgboost_residual_hist.png
│   ├── xgboost_residual_scatter.png
│   ├── xgboost_feature_importance.png
│   └── random_forest_feature_importance.png
├── models/
│   ├── best_model.pkl
│   ├── best_params.json
│   ├── model_metadata.json
│   ├── model_results.csv
│   ├── per_target_metrics.csv
│   ├── hparam_results_xgboost.csv
│   ├── hparam_results_random_forest.csv
│   └── hparam_results_mlp_regressor.csv
├── tune_models.py
├── streamlit_app.py
├── train.py
├── utils.py
└── requirements.txt
```

## Data splitting protocol

The dataset is sorted, so we partition it into **five contiguous blocks** and train on parts 1, 3, and 5 while testing on parts 2 and 4. This mirrors the paper’s emphasis on evaluating interpolation between physiologically different flow regimes without shuffling the chronology.

## Models implemented

| Model | Key settings (after tuning) | Overall RMSE | Overall MAE | Overall R² |
|-------|----------------------------|--------------|-------------|------------|
| **Random Forest** | `n_estimators=500`, `max_depth=16`, `max_features='sqrt'`, `bootstrap=False` | **3.15e-5** | 8.72e-6 | 0.9999998 |
| XGBoost (wrapped by `MultiOutputRegressor`) | `n_estimators=400`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.85` | 5.52e-5 | 2.93e-5 | 0.9999989 |
| MLP Regressor | `hidden_layer_sizes=(256,128)`, `learning_rate_init=5e-4`, `max_iter=650`, batch size 128 | 3.99e-4 | 2.62e-4 | 0.999920 |

- Metrics are computed per target and aggregated globally (RMSE, MAE, MSE, R²).
- Feature scaling is only applied to the neural network branch (StandardScaler).
- Tree-based models expose feature importances; plots live in `figures/`.
- The best overall RMSE belongs to **XGBoost**, which is saved as `models/best_model.pkl` together with metadata for the Streamlit UI.

## Hyperparameter tuning (optional but recommended)

A lightweight sequential tuning routine tries a capped number of configurations per model while preserving the chronological splits:

```bash
cd "/Users/vishnuvardhan_1685/Documents/Projects/Blood Flow Research"
".venv/bin/python" tune_models.py --max-configs 8  # adjust the budget as needed
```

Artifacts saved in `models/` include per-model search tables plus `best_params.json`. When present, `train.py` automatically loads these overrides.

## Training the models

```bash
cd "/Users/vishnuvardhan_1685/Documents/Projects/Blood Flow Research"
".venv/bin/python" -m pip install -r requirements.txt  # once
".venv/bin/python" train.py
```

Outputs include comparison tables (`models/model_results.csv`), per-target metrics, feature importance/residual plots, the correlation heatmap, and ready-to-serve artifacts.

## Running the Streamlit dashboard

```bash
cd "/Users/vishnuvardhan_1685/Documents/Projects/Blood Flow Research"
".venv/bin/python" -m streamlit run streamlit_app.py
```

The dashboard allows manual entry of the eleven control parameters (defaults derived from the training statistics), displays predictions for $(u, u_{plv}, u_p, \tau_w)$, shows overall/per-target RMSE values, surfaces the tuned hyperparameters, and embeds the comparison, residual, and correlation plots. The sidebar also summarizes why KL-Newtonian parameters highlighted in the provided paper influence the selected model.

## Re-using the utilities

- `utils.py` exposes constants (`FEATURE_COLUMNS`, `TARGET_COLUMNS`, `DATA_PATH`), the sequential splitter, reproducible seeding, metric calculators, persistence helpers, and plotting utilities.
- Metadata saved in `models/model_metadata.json` captures best-model metrics plus feature ranges so the Streamlit app can validate user inputs.
- Additional analytics helpers plot the feature/target correlation map, residual histograms, and residual-vs-actual scatter panels for the selected model.

## Next steps

1. Expand the search budget or adopt Bayesian optimizers (Optuna) to probe a richer hyperparameter space while respecting KL-Newtonian constraints.
2. Add sensitivity sliders or tornado charts in Streamlit to visualize how permeability/yield-stress variations shift the wall-shear predictions.
3. Package the training pipeline as a CLI with experiment tracking (MLflow or Weights & Biases) for rapid iteration.
