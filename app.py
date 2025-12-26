# app.py
import streamlit as st
import pandas as pd
import zipfile
from io import BytesIO

from ds_toolkit.preprocessing import apply_preprocessing, missing_summary
from ds_toolkit.feature_select import apply_feature_selection
from ds_toolkit.models import train_models
from ds_toolkit.explain import explain_model
from ds_toolkit.utils import build_standard_report_assets

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------
st.set_page_config(page_title="DataKit Pro", layout="wide")
st.title("⚙️ DataKit Pro — No-Code ML Toolkit")
st.write("Upload → preprocess → feature select → train → explain → export")

# -----------------------------------------------------
# Upload Dataset
# -----------------------------------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------------------------
# Sidebar – Preprocessing
# -----------------------------------------------------
st.sidebar.header("⚙️ Preprocessing")

missing = st.sidebar.selectbox("Missing values", ["none", "mean", "median", "mode", "knn"])
outliers = st.sidebar.selectbox("Outliers", ["none", "iqr", "zscore"])
encoding = st.sidebar.selectbox("Encoding", ["none", "onehot", "ordinal"])
scaling = st.sidebar.selectbox("Scaling", ["none", "standard", "minmax"])

# -----------------------------------------------------
# Target Selection
# -----------------------------------------------------
st.subheader("🎯 Target Column")
target_col = st.selectbox("Select target", df.columns)

if st.checkbox("Show missing summary"):
    st.write(missing_summary(df))

# -----------------------------------------------------
# Apply Preprocessing
# -----------------------------------------------------
if st.button("Apply Preprocessing"):
    df_pre, scaler = apply_preprocessing(
        df,
        missing=missing,
        outliers=outliers,
        encoding=encoding,
        scaling=scaling,
        target_col=target_col
    )
    st.session_state["df_pre"] = df_pre
    st.success("Preprocessing completed")
    st.dataframe(df_pre.head())

# -----------------------------------------------------
# Feature Selection
# -----------------------------------------------------
if "df_pre" in st.session_state:
    st.subheader("🧬 Feature Selection")

    method = st.selectbox("Method", ["none", "kbest", "rfe", "corr"])
    k = st.number_input("K (if applicable)", min_value=1, value=10)

    if st.button("Select Features"):
        df_pre = st.session_state["df_pre"]
        X = df_pre.drop(columns=[target_col])
        y = df_pre[target_col]

        X_new, cols = apply_feature_selection(X, y, method=method, k=k)
        st.session_state["X_new"] = X_new
        st.session_state["y"] = y

        st.write("Selected features:", cols)
        st.dataframe(X_new.head())

# -----------------------------------------------------
# Model Training
# -----------------------------------------------------
if "X_new" in st.session_state:
    st.subheader("🤖 Model Training")

    models_to_run = st.multiselect(
        "Choose models",
        [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "linear_regression",
            "decision_tree_regressor",
            "random_forest_regressor",
        ]
    )

    test_size = st.slider("Test size (%)", 10, 40, 20) / 100
    use_cv = st.checkbox("Use Cross-Validation")
    cv_folds = st.number_input("CV folds", min_value=2, value=5)

    if st.button("Train Models"):
        results = train_models(
            st.session_state["X_new"],
            st.session_state["y"],
            models_to_run=models_to_run,
            test_size=test_size,
            use_cv=use_cv,
            cv_folds=cv_folds,
        )
        st.session_state["results"] = results
        st.success("Training complete")
        st.json(results["metrics"])

# -----------------------------------------------------
# SHAP Explainability (FIXED)
# -----------------------------------------------------
if "results" in st.session_state:
    st.subheader("🔍 SHAP Explainability")

    results = st.session_state["results"]
    model_names = list(results["models"].keys())

    if model_names:
        model_name = st.selectbox("Select model", model_names)
        model = results["models"][model_name]
        X_new = st.session_state["X_new"]

        with st.spinner("Computing SHAP..."):
            explain_output = explain_model(model, X_new)

        if explain_output:
            st.session_state["explain_output"] = explain_output
            st.pyplot(explain_output["summary_fig"])
            st.pyplot(explain_output["waterfall_fig"])
        else:
            st.warning("SHAP not supported for this model.")

# -----------------------------------------------------
# Export Report
# -----------------------------------------------------
if "results" in st.session_state:
    st.subheader("📦 Export Report")

    metrics = st.session_state["results"]["metrics"]
    images = {}

    explain_output = st.session_state.get("explain_output")

    if explain_output:
        for key in ["summary_fig", "waterfall_fig"]:
            buf = BytesIO()
            explain_output[key].savefig(buf, format="png")
            images[f"{key}.png"] = buf.getvalue()

    cleaned_csv = st.session_state["df_pre"].to_csv(index=False).encode()

    entries = build_standard_report_assets(
        metrics=metrics,
        images=images,
        cleaned_csv_bytes=cleaned_csv
    )

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        for e in entries:
            z.writestr(e["name"], e["bytes"])

    st.download_button(
        "Download Report (ZIP)",
        zip_buffer.getvalue(),
        file_name="datakit_pro_report.zip",
        mime="application/zip"
    )
