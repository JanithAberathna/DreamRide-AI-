"""Shared data loading, cleaning, feature engineering, and preprocessing utilities
for Lanka AutoPrice AI (Sri Lankan Vehicle Price Prediction).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "car_price_dataset.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

REFERENCE_YEAR = 2025
FACILITY_COLS = ["AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"]
NUMERICAL_FEATURES = ["Vehicle_Age", "Engine (cc)", "Facility_Count"] + FACILITY_COLS
OHE_CATEGORICAL_FEATURES = ["Brand", "Gear", "Fuel Type", "Town", "Leasing", "Condition"]
TARGET_ENC_FEATURES = ["Model", "Brand_Model"]
ALL_MODEL_FEATURES = NUMERICAL_FEATURES + OHE_CATEGORICAL_FEATURES + TARGET_ENC_FEATURES
TARGET_COL = "Price"


def load_raw_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load the raw Sri Lankan vehicle CSV file."""
    return pd.read_csv(path)


def audit_dataset(raw_df: pd.DataFrame) -> dict:
    """Compute data quality audit statistics for documentation & dashboard display."""
    no_idx = raw_df.drop(columns=["Unnamed: 0"], errors="ignore")
    no_date = no_idx.drop(columns=["Date"], errors="ignore")
    expected_mileage = (REFERENCE_YEAR - raw_df["YOM"]) * 11000
    synthetic_match_pct = float(((raw_df["Millage(KM)"] - expected_mileage).abs() == 0).mean() * 100)
    yom_mileage_corr = float(raw_df["YOM"].corr(raw_df["Millage(KM)"]))

    return {
        "raw_rows": int(len(raw_df)),
        "raw_columns": int(raw_df.shape[1]),
        "missing_values": int(raw_df.isna().sum().sum()),
        "exact_duplicates_all_cols": int(raw_df.duplicated().sum()),
        "exact_duplicates_excl_index": int(no_idx.duplicated().sum()),
        "cross_date_reposts_excl_date": int(no_date.duplicated().sum()),
        "synthetic_mileage_match_pct": round(synthetic_match_pct, 2),
        "yom_mileage_correlation": round(yom_mileage_corr, 6),
        "date_min": str(raw_df["Date"].min()),
        "date_max": str(raw_df["Date"].max()),
        "unique_brands": int(raw_df["Brand"].nunique()),
        "unique_models": int(raw_df["Model"].nunique()),
        "unique_towns": int(raw_df["Town"].nunique()),
        "price_min_lakhs": float(raw_df["Price"].min()),
        "price_median_lakhs": float(raw_df["Price"].median()),
        "price_mean_lakhs": round(float(raw_df["Price"].mean()), 2),
        "price_p99_lakhs": round(float(raw_df["Price"].quantile(0.99)), 2),
        "price_max_lakhs": float(raw_df["Price"].max()),
        "yom_min": int(raw_df["YOM"].min()),
        "yom_max": int(raw_df["YOM"].max()),
    }


def clean_and_engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean identifiers/duplicates and engineer domain features strictly available at prediction time.

    Key design decisions:
    1. Drop `Unnamed: 0` (row index identifier).
    2. Remove exact duplicate listings (identical features & advertisement date).
    3. Create `Vehicle_Age = 2025 - YOM` (using dataset reference year 2025 rather than exact
       advertisement date so predictions never depend on listing dates).
    4. Drop `Millage(KM)` from modeling features because audit proved Millage(KM) == (2025 - YOM)*11000
       for 100% of rows (r = -1.0000), causing exact multicollinearity with Vehicle_Age.
    5. Encode vehicle facilities as binary (1 = Available, 0 = Not_Available) and compute `Facility_Count`.
    6. Create `Brand_Model` composite feature to disambiguate identical model names across brands.
    7. Retain `Date_dt` separately ONLY for chronological split robustness validation (never as an ML feature).
    """
    df = raw_df.drop(columns=["Unnamed: 0"], errors="ignore").copy()

    # Standardize string columns defensively
    for col in ["Brand", "Model", "Gear", "Fuel Type", "Town", "Leasing", "Condition"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Remove exact duplicate rows (excluding Unnamed: 0)
    df = df.drop_duplicates().reset_index(drop=True)

    # Parse Date strictly for EDA and chronological validation split
    if "Date" in df.columns:
        df["Date_dt"] = pd.to_datetime(df["Date"])

    # Engineer Vehicle_Age from YOM using reference year 2025
    df["Vehicle_Age"] = (REFERENCE_YEAR - df["YOM"]).clip(lower=0)

    # Binary encode vehicle facilities
    for col in FACILITY_COLS:
        if df[col].dtype == object:
            df[col] = (df[col].str.strip() == "Available").astype(int)
        else:
            df[col] = df[col].astype(int)

    df["Facility_Count"] = df[FACILITY_COLS].sum(axis=1)
    df["Brand_Model"] = df["Brand"] + " " + df["Model"]

    return df


def build_feature_row(
    brand: str,
    model: str,
    yom: int,
    engine_cc: float,
    gear: str,
    fuel_type: str,
    town: str,
    leasing: str,
    condition: str,
    air_condition: bool,
    power_steering: bool,
    power_mirror: bool,
    power_window: bool,
) -> pd.DataFrame:
    """Create a single-row DataFrame matching the exact model feature schema for live prediction."""
    ac = int(air_condition)
    ps = int(power_steering)
    pm = int(power_mirror)
    pw = int(power_window)
    row = {
        "Vehicle_Age": max(0, REFERENCE_YEAR - int(yom)),
        "Engine (cc)": float(engine_cc),
        "Facility_Count": ac + ps + pm + pw,
        "AIR CONDITION": ac,
        "POWER STEERING": ps,
        "POWER MIRROR": pm,
        "POWER WINDOW": pw,
        "Brand": brand.strip(),
        "Gear": gear.strip(),
        "Fuel Type": fuel_type.strip(),
        "Town": town.strip(),
        "Leasing": leasing.strip(),
        "Condition": condition.strip(),
        "Model": model.strip(),
        "Brand_Model": f"{brand.strip()} {model.strip()}",
    }
    return pd.DataFrame([row])[ALL_MODEL_FEATURES]


def build_preprocessor(random_state: int = 42) -> ColumnTransformer:
    """Create a leak-free ColumnTransformer combining:
    - StandardScaler for numeric/binary features
    - OneHotEncoder (handle_unknown='ignore', min_frequency=3) for Brand, Town, Gear, Fuel Type, Leasing, Condition
    - Cross-validated TargetEncoder (cv=5, smooth='auto') for high-cardinality Model & Brand_Model (1,555 models)
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat_ohe",
                OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse_output=False),
                OHE_CATEGORICAL_FEATURES,
            ),
            (
                "cat_te",
                TargetEncoder(smooth="auto", cv=5, random_state=random_state),
                TARGET_ENC_FEATURES,
            ),
        ],
        remainder="drop",
    )


def format_lkr_lakhs(val_lakhs: float) -> str:
    """Format a price value in LKR Lakhs cleanly."""
    return f"Rs. {val_lakhs:,.2f} Lakhs"


def format_lkr_millions(val_lakhs: float) -> str:
    """Convert LKR Lakhs (1 Lakh = 0.1 Million LKR) to LKR Millions string."""
    return f"LKR {val_lakhs / 10.0:,.2f}M"
