"""End-to-end training, cross-validation, robustness evaluation, explainability,
and artifact persistence pipeline for Lanka AutoPrice AI.

Models evaluated on identical 80/20 split (random_state=42) and 5-fold CV:
  1. Baseline : Linear Regression
  2. Bagging  : Random Forest Regressor
  3. Boosting : Gradient Boosting Regressor
  4. Stacking : Stacking Regressor (Random Forest + Gradient Boosting -> RidgeCV meta-model)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data_utils import (
    ALL_MODEL_FEATURES,
    ARTIFACTS_DIR,
    DATASET_PATH,
    TARGET_COL,
    audit_dataset,
    build_preprocessor,
    clean_and_engineer_features,
    load_raw_dataset,
)

RANDOM_STATE = 42


def make_model_pipelines(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Construct the four model pipelines sharing an identical preprocessing design."""
    rf_params = dict(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    gbr_params = dict(
        n_estimators=350,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.8,
        random_state=random_state,
    )

    # Note: inside StackingRegressor, n_jobs=1 on outer stacking avoids thread oversubscription
    # with inner RandomForestRegressor(n_jobs=-1), making training ~4x faster and deterministic.
    stack_rf_params = {**rf_params}
    stack_gbr_params = {**gbr_params}

    models = {
        "Linear Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(random_state)),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(random_state)),
                ("regressor", RandomForestRegressor(**rf_params)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor(random_state)),
                ("regressor", GradientBoostingRegressor(**gbr_params)),
            ]
        ),
        "Stacking Regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor(random_state)),
                (
                    "regressor",
                    StackingRegressor(
                        estimators=[
                            ("rf", RandomForestRegressor(**stack_rf_params)),
                            ("gbr", GradientBoostingRegressor(**stack_gbr_params)),
                        ],
                        final_estimator=RidgeCV(alphas=np.logspace(-2, 3, 25)),
                        cv=5,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return regression metrics (MAE, RMSE, R2, MedAE, MAPE) with non-negative clipping."""
    y_pred_clipped = np.clip(y_pred, 0.0, None)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(root_mean_squared_error(y_true, y_pred_clipped))
    r2 = float(r2_score(y_true, y_pred_clipped))
    medae = float(np.median(np.abs(y_true - y_pred_clipped)))
    mape = float(np.mean(np.abs((y_true - y_pred_clipped) / np.clip(y_true, 1e-6, None))) * 100.0)
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MedAE": round(medae, 4),
        "MAPE_pct": round(mape, 2),
    }


def evaluate_split_strategy(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    strategy_name: str,
) -> dict[str, dict[str, float]]:
    """Train and evaluate all 4 models on a given train/test index split."""
    X = df[ALL_MODEL_FEATURES]
    y = df[TARGET_COL].values
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    pipelines = make_model_pipelines(RANDOM_STATE)
    results = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        results[name] = compute_metrics(y_te, preds)
    print(f"  Completed split robustness experiment: {strategy_name} (train={len(train_idx)}, test={len(test_idx)})")
    return results


def build_dropdown_metadata(df: pd.DataFrame) -> dict:
    """Create rich UI metadata for the Streamlit Vehicle Price Predictor."""
    brands = sorted(df["Brand"].unique().tolist())
    brand_to_models: dict[str, list[str]] = {}
    model_defaults: dict[str, dict] = {}

    for brand, grp in df.groupby("Brand"):
        # Sort models by listing frequency descending for convenient selection
        models_sorted = grp["Model"].value_counts().index.tolist()
        brand_to_models[brand] = models_sorted

    for (brand, model), grp in df.groupby(["Brand", "Model"]):
        key = f"{brand}||{model}"
        model_defaults[key] = {
            "median_yom": int(round(grp["YOM"].median())),
            "median_engine": float(round(grp["Engine (cc)"].median())),
            "mode_gear": str(grp["Gear"].mode().iloc[0]),
            "mode_fuel": str(grp["Fuel Type"].mode().iloc[0]),
            "listing_count": int(len(grp)),
            "median_price": round(float(grp["Price"].median()), 2),
        }

    return {
        "brands": brands,
        "brand_to_models": brand_to_models,
        "model_defaults": model_defaults,
        "towns": df["Town"].value_counts().index.tolist(),
        "gears": sorted(df["Gear"].unique().tolist()),
        "fuel_types": sorted(df["Fuel Type"].unique().tolist()),
        "leasing_options": sorted(df["Leasing"].unique().tolist()),
        "conditions": sorted(df["Condition"].unique().tolist()),
        "yom_min": int(df["YOM"].min()),
        "yom_max": int(df["YOM"].max()),
        "engine_min": float(df["Engine (cc)"].min()),
        "engine_max": float(df["Engine (cc)"].max()),
    }


def run_training_pipeline(
    dataset_path: str | Path = DATASET_PATH,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
) -> dict:
    """Execute the full Lanka AutoPrice AI experiment and save all artifacts."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading and auditing Sri Lankan vehicle dataset...")
    raw_df = load_raw_dataset(dataset_path)
    audit_stats = audit_dataset(raw_df)
    df = clean_and_engineer_features(raw_df)
    audit_stats["clean_rows"] = int(len(df))
    audit_stats["duplicates_removed"] = audit_stats["raw_rows"] - audit_stats["clean_rows"]

    X = df[ALL_MODEL_FEATURES]
    y = df[TARGET_COL].values

    print("[2/6] Training all 4 models on identical 80/20 train/test split (random_state=42)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.2, random_state=RANDOM_STATE
    )

    pipelines = make_model_pipelines(RANDOM_STATE)
    test_metrics: dict[str, dict] = {}
    train_metrics: dict[str, dict] = {}
    train_times: dict[str, float] = {}
    test_pred_df = df.iloc[idx_test][
        ["Brand", "Model", "YOM", "Vehicle_Age", "Engine (cc)", "Gear", "Fuel Type", "Town", "Condition", "Leasing", "Price"]
    ].copy()

    model_filenames = {
        "Linear Regression": "linear_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "Stacking Regressor": "stacking_regressor.joblib",
    }

    fitted_pipelines: dict[str, Pipeline] = {}
    stacking_meta_info: dict = {}

    for name, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)
        train_times[name] = elapsed

        train_preds = np.clip(pipe.predict(X_train), 0.0, None)
        test_preds = np.clip(pipe.predict(X_test), 0.0, None)

        train_metrics[name] = compute_metrics(y_train, train_preds)
        test_metrics[name] = compute_metrics(y_test, test_preds)
        test_pred_df[f"Pred_{name}"] = np.round(test_preds, 3)
        fitted_pipelines[name] = pipe

        joblib.dump(pipe, artifacts_dir / model_filenames[name], compress=3)
        print(
            f"  {name:20s} | Test MAE={test_metrics[name]['MAE']:6.2f} | "
            f"RMSE={test_metrics[name]['RMSE']:6.2f} | R2={test_metrics[name]['R2']:.4f} | {elapsed:.1f}s"
        )

        if name == "Stacking Regressor":
            meta = pipe.named_steps["regressor"].final_estimator_
            stacking_meta_info = {
                "meta_model": "RidgeCV",
                "selected_alpha": round(float(meta.alpha_), 4),
                "intercept": round(float(meta.intercept_), 4),
                "base_weights": {
                    "Random Forest": round(float(meta.coef_[0]), 4),
                    "Gradient Boosting": round(float(meta.coef_[1]), 4),
                },
            }
            print(f"  Stacking Meta-Learner Weights: {stacking_meta_info}")

    print("[3/6] Running 5-fold Cross-Validation across all 4 models...")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    cv_results: dict[str, dict] = {}
    cv_pipelines = make_model_pipelines(RANDOM_STATE)
    for name, pipe in cv_pipelines.items():
        t0 = time.time()
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
        mae_folds = -scores["test_mae"]
        rmse_folds = -scores["test_rmse"]
        r2_folds = scores["test_r2"]
        cv_results[name] = {
            "MAE_mean": round(float(mae_folds.mean()), 4),
            "MAE_std": round(float(mae_folds.std()), 4),
            "RMSE_mean": round(float(rmse_folds.mean()), 4),
            "RMSE_std": round(float(rmse_folds.std()), 4),
            "R2_mean": round(float(r2_folds.mean()), 4),
            "R2_std": round(float(r2_folds.std()), 4),
            "MAE_folds": [round(float(v), 4) for v in mae_folds],
            "RMSE_folds": [round(float(v), 4) for v in rmse_folds],
            "R2_folds": [round(float(v), 4) for v in r2_folds],
            "cv_time_sec": round(time.time() - t0, 2),
        }
        print(
            f"  5-Fold CV {name:20s} | MAE={cv_results[name]['MAE_mean']:.2f}±{cv_results[name]['MAE_std']:.2f} | "
            f"RMSE={cv_results[name]['RMSE_mean']:.2f}±{cv_results[name]['RMSE_std']:.2f} | "
            f"R2={cv_results[name]['R2_mean']:.4f}±{cv_results[name]['R2_std']:.4f}"
        )

    print("[4/6] Running Split Robustness Experiments (Chronological & Grouped by Brand+Model)...")
    # Chronological Split: earliest 80% listings train, latest 20% listings test
    chrono_order = df.sort_values("Date_dt").index.values
    split_point = int(len(chrono_order) * 0.8)
    chrono_train_idx = chrono_order[:split_point]
    chrono_test_idx = chrono_order[split_point:]
    chrono_cutoff_date = str(df.loc[chrono_test_idx, "Date_dt"].min().date())

    # Grouped Split: GroupShuffleSplit by Brand_Model so exact models are never in both train & test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    grp_train_idx, grp_test_idx = next(gss.split(X, y, groups=df["Brand_Model"]))

    split_robustness = {
        "random_80_20": {
            "description": "Standard 80/20 random split (random_state=42)",
            "train_size": int(len(idx_train)),
            "test_size": int(len(idx_test)),
            "metrics": test_metrics,
        },
        "chronological_80_20": {
            "description": f"Chronological split (test >= {chrono_cutoff_date}; verifies no temporal leakage)",
            "cutoff_date": chrono_cutoff_date,
            "train_size": int(len(chrono_train_idx)),
            "test_size": int(len(chrono_test_idx)),
            "metrics": evaluate_split_strategy(df, chrono_train_idx, chrono_test_idx, "Chronological 80/20"),
        },
        "grouped_brand_model": {
            "description": "GroupShuffleSplit by Brand+Model (evaluates generalization to unseen vehicle models)",
            "train_size": int(len(grp_train_idx)),
            "test_size": int(len(grp_test_idx)),
            "metrics": evaluate_split_strategy(df, grp_train_idx, grp_test_idx, "Grouped by Brand+Model"),
        },
    }

    print("[5/6] Computing Permutation Feature Importance on held-out test set...")
    perm_rows = []
    for name in ["Random Forest", "Gradient Boosting", "Stacking Regressor"]:
        perm = permutation_importance(
            fitted_pipelines[name],
            X_test,
            y_test,
            n_repeats=8,
            random_state=RANDOM_STATE,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        for feat, imp_mean, imp_std in zip(ALL_MODEL_FEATURES, perm.importances_mean, perm.importances_std):
            perm_rows.append(
                {
                    "Model": name,
                    "Feature": feat,
                    "RMSE_Increase_Mean": round(float(imp_mean), 4),
                    "RMSE_Increase_Std": round(float(imp_std), 4),
                }
            )
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    print("[6/6] Saving evaluation summary, test predictions, and UI metadata...")
    test_pred_df.to_csv(artifacts_dir / "test_predictions.csv", index=False)

    dropdown_meta = build_dropdown_metadata(df)
    with open(artifacts_dir / "dropdown_metadata.json", "w", encoding="utf-8") as f:
        json.dump(dropdown_meta, f, indent=2)

    summary = {
        "project": "Lanka AutoPrice AI — Sri Lankan Vehicle Price Prediction Using Ensemble Learning",
        "price_unit": "LKR Lakhs (1 Lakh = Rs. 100,000 = LKR 0.1 Million)",
        "dataset_source": "Kaggle — SriLankan Vehicle(Car) Price Dataset (scraped from Sri Lankan marketplaces, Dec 2024 – Feb 2025)",
        "dataset_audit": audit_stats,
        "features": {
            "numerical": ALL_MODEL_FEATURES[:7],
            "categorical_ohe": ALL_MODEL_FEATURES[7:13],
            "categorical_target_encoded": ALL_MODEL_FEATURES[13:],
            "excluded_features": {
                "Unnamed: 0": "Row index identifier (non-predictive)",
                "Date": "Advertisement date — unavailable at inference time (used strictly for chronological split validation)",
                "Millage(KM)": "100% deterministically equal to (2025 - YOM) * 11,000 (r = -1.0000); dropped to prevent exact multicollinearity with Vehicle_Age",
            },
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_times_sec": train_times,
        "cv_5fold": cv_results,
        "stacking_meta_learner": stacking_meta_info,
        "split_robustness": split_robustness,
    }

    with open(artifacts_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll models and evaluation artifacts saved to: {artifacts_dir}")
    return summary


if __name__ == "__main__":
    run_training_pipeline()
