"""DreamRide AI — Sri Lankan Vehicle Price Valuation & Ensemble Analytics Suite (Streamlit)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_utils import (
    ARTIFACTS_DIR as DEFAULT_ARTIFACTS_DIR,
    DATASET_PATH,
    clean_and_engineer_features,
    load_raw_dataset,
)
from src.ui_sections import (
    render_eda_tab,
    render_ensemble_tab,
    render_explainability_tab,
    render_overview_tab,
    render_predictor_tab,
)
from src.ui_theme import inject_editorial_css, render_masthead

ARTIFACTS_DIR = Path(os.environ.get("LANKA_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))

MODEL_FILES = {
    "Stacking Regressor": "stacking_regressor.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
    "Linear Regression": "linear_regression.joblib",
}

st.set_page_config(
    page_title="DreamRide AI — Sri Lankan Vehicle Valuation Suite",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_editorial_css()


# ---------------------------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ensemble pipelines...")
def load_models() -> dict:
    return {name: joblib.load(ARTIFACTS_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data(show_spinner=False)
def load_dropdown_metadata() -> dict:
    with open(ARTIFACTS_DIR / "dropdown_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_evaluation_summary() -> dict | None:
    path = ARTIFACTS_DIR / "evaluation_summary.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame | None:
    path = ARTIFACTS_DIR / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame | None:
    path = ARTIFACTS_DIR / "test_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    return clean_and_engineer_features(load_raw_dataset(DATASET_PATH))


def artifacts_ready() -> bool:
    required = [ARTIFACTS_DIR / "dropdown_metadata.json"] + [
        ARTIFACTS_DIR / fname for fname in MODEL_FILES.values()
    ]
    return all(p.exists() for p in required)


# ---------------------------------------------------------------------------
# Artifact Guard
# ---------------------------------------------------------------------------
if not artifacts_ready():
    st.warning("Trained models were not found in `artifacts/`. Train the ensemble pipeline to launch the dashboard.")
    if st.button("Train Models Now", type="primary"):
        from src.train_pipeline import run_training_pipeline

        with st.spinner("Training Linear Regression, Random Forest, Gradient Boosting, and Stacking Regressor..."):
            run_training_pipeline(DATASET_PATH, ARTIFACTS_DIR)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.stop()

models = load_models()
meta = load_dropdown_metadata()
summary = load_evaluation_summary()
df = load_clean_data()
fi_df = load_feature_importance()
preds_df = load_test_predictions()

best_r2 = summary["test_metrics"]["Stacking Regressor"]["R2"] if summary else 0.8507
best_mae = summary["test_metrics"]["Stacking Regressor"]["MAE"] if summary else 5.57

# ---------------------------------------------------------------------------
# Editorial Showroom Masthead & Primary Navigation
# ---------------------------------------------------------------------------
render_masthead(
    n_listings=len(df),
    n_brands=len(meta["brands"]),
    best_r2=best_r2,
    best_mae=best_mae,
)

tab_pred, tab_over, tab_eda, tab_ens, tab_exp = st.tabs(
    [
        "Valuation Studio",
        "Executive Overview",
        "Market Explorer (EDA)",
        "Ensemble & Stacking Lab",
        "Explainability & Diagnostics",
    ]
)

with tab_pred:
    render_predictor_tab(models, meta, df, summary)

with tab_over:
    render_overview_tab(summary)

with tab_eda:
    render_eda_tab(df)

with tab_ens:
    render_ensemble_tab(summary)

with tab_exp:
    render_explainability_tab(fi_df, preds_df)
