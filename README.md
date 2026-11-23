# DreamRide AI — Sri Lankan Vehicle Valuation & Ensemble Analytics Suite

An **Ensemble Learning** project and editorial-grade **Streamlit** analytics suite for predicting Sri Lankan vehicle prices (in **LKR Lakhs**, where `1 Lakh = Rs. 100,000 = LKR 0.1M`) using **Linear Regression (Baseline)**, **Random Forest (Bagging)**, **Gradient Boosting (Boosting)**, and **Stacking Regressor (Hybrid)**.

---

## Key Highlights

- **Dataset & Forensic Data Audit**: 9,788 raw Sri Lankan marketplace listings (Dec 2024 – Feb 2025) covering **50 brands**, **1,555 models**, and **107 towns** with zero missing values.
  - Removed **18 exact duplicate listings** → **9,770 audited vehicles** (prices from Rs. 2.65L to Rs. 790.00L; median Rs. 43.00L).
  - Identified that `Millage(KM)` is 100% synthetically computed as `(2025 - YOM) × 11,000` (`r = -1.0000`) and excluded it to prevent exact multicollinearity with engineered `Vehicle_Age`.
  - Excluded `Unnamed: 0` and listing `Date` from model features (`Date` is retained solely for chronological split validation).
- **Leak-Free Preprocessing Pipeline**: Single scikit-learn `ColumnTransformer` combining `StandardScaler` (numeric & binary facility indicators), `OneHotEncoder(min_frequency=3)` (`Brand`, `Gear`, `Fuel Type`, `Town`, `Leasing`, `Condition`), and 5-fold cross-validated `TargetEncoder` (`Model`, `Brand_Model`).
- **Stacking Hybridization**: Combines out-of-fold predictions from **Random Forest** (variance reduction) and **Gradient Boosting** (bias reduction) via a regularized `RidgeCV` meta-learner (`alpha = 1000.0`, weights: **Random Forest `0.5327`**, **Gradient Boosting `0.4844`**, intercept `-0.8874`).

---

## Model Performance Summary

All models are trained and evaluated on the identical 80/20 split (`train = 7,816`, `test = 1,954`, `random_state = 42`). Errors are reported in **LKR Lakhs**.

### Holdout Test Set (80/20 Random Split)

| Model | Family | MAE (Lakhs) | RMSE (Lakhs) | R² | MedAE (Lakhs) | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | Baseline | 11.43 | 27.16 | 0.6894 | 6.78 | 28.59% |
| Random Forest | Bagging | 5.66 | 19.33 | 0.8427 | **2.15** | **13.11%** |
| Gradient Boosting | Boosting | 6.03 | 19.25 | 0.8441 | 2.76 | 15.52% |
| **Stacking Regressor** | **Hybrid** | **5.57** | **18.83** | **0.8507** | 2.32 | 13.55% |

### 5-Fold Cross-Validation & Split Robustness

| Validation Strategy | Random Forest (R² / MAE) | Gradient Boosting (R² / MAE) | Stacking Regressor (R² / MAE) |
| :--- | :---: | :---: | :---: |
| **5-Fold Cross-Validation** | `0.8927 ± 0.036` / `5.10L` | `0.8958 ± 0.033` / `5.55L` | `0.8883 ± 0.028` / `5.93L` |
| **Chronological Split** (`test >= 2025-02-02`) | `0.8885` / `5.26L` | `0.9100` / `5.62L` | `0.9077` / `5.27L` |
| **Grouped Split** (Unseen `Brand + Model`) | `0.8190` / `10.06L` | `0.8255` / `10.24L` | `0.7913` / `11.45L` |

---

## Interactive Dashboard Views

1. **Valuation Studio** — Live multi-model inference across all 4 pipelines, showroom presets, Fair Market Range, Deal Analyzer (*"Is this listing overpriced?"*), Year-of-Manufacture depreciation curve, and real comparable market listings.
2. **Executive Overview** — KPI highlights, end-to-end leak-free ML pipeline architecture, and forensic data audit findings.
3. **Market Explorer (EDA)** — Interactive filtering across 9,770 audited listings with price distributions, top brands, depreciation trajectories, fuel/transmission spreads, and regional town benchmarks.
4. **Ensemble & Stacking Lab** — Holdout leaderboard, 5-fold cross-validation stability, `RidgeCV` meta-learner weights, and Split Robustness comparison (Random vs. Chronological vs. Unseen Brand+Model).
5. **Explainability & Diagnostics** — Permutation feature importance, Actual vs. Predicted parity scatter, residual distributions, accuracy by price tier, and closest predictions vs. luxury edge cases.

---

## Project Structure

```text
DreamRide AI/
├── .streamlit/
│   └── config.toml                   # Editorial Light theme configuration
├── artifacts/                        # Pre-trained .joblib pipelines & evaluation outputs
│   ├── stacking_regressor.joblib     # Hybrid Stacking Regressor pipeline
│   ├── random_forest.joblib          # Random Forest (Bagging) pipeline
│   ├── gradient_boosting.joblib      # Gradient Boosting (Boosting) pipeline
│   ├── linear_regression.joblib      # Linear Regression (Baseline) pipeline
│   ├── dropdown_metadata.json        # Brand/model dropdowns & showroom defaults
│   ├── evaluation_summary.json       # Holdout, 5-fold CV & split robustness metrics
│   ├── feature_importance.csv        # Permutation feature importance scores
│   └── test_predictions.csv          # Held-out test predictions & residuals
├── dataset/
│   └── car_price_dataset.csv         # Sri Lankan vehicle listings dataset (9,788 rows)
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── data_utils.py                 # Cleaning, audit, feature engineering & preprocessing
│   ├── train_pipeline.py             # End-to-end training, CV, robustness & artifact export
│   ├── ui_theme.py                   # Editorial Light CSS, Plotly styling & KPI components
│   └── ui_sections.py                # 5 interactive Streamlit dashboard views
├── app.py                            # Multi-tab Streamlit valuation & analytics suite
├── DreamRide AI.ipynb                # Jupyter Notebook with full analysis & experiments
├── PRESENTATION_SCRIPT.md            # 1.5-minute presentation script & AI usage declaration
└── README.md                         # Project documentation
```

---

## Quick Start

### 1. Install Dependencies
Requires Python 3.10+ (scikit-learn `>= 1.4` for `TargetEncoder` and `root_mean_squared_error`):

```bash
pip install streamlit scikit-learn pandas numpy plotly joblib
```

### 2. Launch the Streamlit Dashboard
Pre-trained models and evaluation metadata are already included in `artifacts/`:

```bash
streamlit run app.py
```

### 3. Retrain Models & Regenerate Artifacts *(Optional)*
To re-run the full training, 5-fold cross-validation, split robustness checks, and permutation importance pipeline from scratch:

```bash
python -m src.train_pipeline
```

*(If `artifacts/` is deleted, launching `app.py` also provides a built-in **Train Models Now** button to regenerate everything automatically.)*
