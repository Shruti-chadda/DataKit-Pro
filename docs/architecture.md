DataKit Pro — Project Architecture

1. Overview

DataKit Pro is a no-code, Streamlit-based machine learning workflow tool that allows users to upload a CSV, explore data, preprocess it, run feature selection, train ML models, view evaluation metrics, and generate SHAP-based explainability reports.

2. System Architecture
   UI Layer (Streamlit – app.py)

Handles all user interactions (file upload, parameter selection, buttons).

Displays previews, charts, metrics, SHAP plots, and export options.

Uses st.session_state to store dataset, pipeline, and model results.

Core Logic Layer (ds_toolkit/)

Contains all reusable Python modules:

File Responsibility
preprocessing.py Missing values, outliers, encoding, scaling
feature_select.py SelectKBest, RFE, correlation-based selection
models.py Train/test split, model wrappers, evaluation
explain.py SHAP explainability helpers
utils.py Load/save, pipeline config, report helpers 3. Data Flow (Step-by-Step)

User uploads CSV → loaded with pandas.

EDA → missing summary, statistics, correlation heatmap.

Preprocessing → imputation → outliers → encoding → scaling.

Feature Selection → user chooses SelectKBest or RFE.

Model Training → Logistic Regression, Random Forest, etc.

Evaluation → metrics + confusion matrix / regression plots.

Explainability → SHAP global & local importance.

Export → cleaned dataset, saved model, and report.

4. Key Design Principles

Modular: GUI separate from ML logic.

Reproducible: fixed random seed, saved pipeline config.

Extendable: easy to add new models or preprocessing options.

User-friendly: no backend or database required.
