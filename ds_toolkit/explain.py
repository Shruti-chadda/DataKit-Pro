"""
ds_toolkit/explain.py

SHAP-based model explainability helpers for DataKit Pro.
Provides:
- get_shap_explainer(...)
- compute_shap_values(...)
- shap_summary_plot(...)
- shap_waterfall_plot(...)
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Create SHAP Explainer
# ---------------------------------------------------------
def get_shap_explainer(model, X_sample: pd.DataFrame):
    """
    Returns an appropriate SHAP explainer based on model type.
    Uses:
      - TreeExplainer for tree-based models
      - LinearExplainer for linear models
      - KernelExplainer fallback for others
    """
    try:
        # Tree-based models
        if hasattr(model, "estimators_"):
            return shap.TreeExplainer(model, feature_names=X_sample.columns.tolist())
        # Linear models
        if model.__class__.__name__.lower().startswith("logistic") or \
           model.__class__.__name__.lower().startswith("linear"):
            return shap.LinearExplainer(model, X_sample)
        # Fallback
        return shap.KernelExplainer(model.predict, X_sample)
    except Exception:
        # worst-case fallback
        return shap.KernelExplainer(model.predict, X_sample)


# ---------------------------------------------------------
# Compute SHAP values
# ---------------------------------------------------------
def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 200):
    """
    Computes SHAP values for a model using a sample of X (to avoid heavy computation).
    Returns explainer, shap_values, and X_sample (subset used).
    """
    # sample to avoid computation overload
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X.copy()

    explainer = get_shap_explainer(model, X_sample)

    try:
        shap_values = explainer(X_sample)
    except Exception:
        # fallback for kernel explainer (older SHAP versions)
        shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values, X_sample


# ---------------------------------------------------------
# Global Summary Plot (Feature Importance)
# ---------------------------------------------------------
def shap_summary_plot(shap_values, X_sample: pd.DataFrame):
    """
    Generates a SHAP global feature importance summary plot.
    Returns a matplotlib figure object that Streamlit can render.
    """
    fig = plt.figure(figsize=(8, 6))
    try:
        shap.summary_plot(shap_values, X_sample, show=False, plot_size=(8, 6))
    except Exception:
        shap.summary_plot(shap_values.values, X_sample, show=False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# Single Prediction Explanation (Waterfall plot)
# ---------------------------------------------------------
def shap_waterfall_plot(shap_values, X_sample: pd.DataFrame, index: int = 0):
    """
    Generates a SHAP waterfall plot for a single row of data.
    By default explains row 0 of the sample.
    """
    # pick one instance
    if index < 0 or index >= len(X_sample):
        index = 0

    fig = plt.figure(figsize=(8, 6))

    try:
        shap.plots.waterfall(shap_values[index], show=False)
    except Exception:
        # fallback to force plot saved as PNG if waterfall not supported
        shap.force_plot(
            shap_values[index].base_values,
            shap_values[index].values,
            X_sample.iloc[index],
            matplotlib=True,
            show=False,
        )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# Unified Explainability Helper
# ---------------------------------------------------------
def explain_model(model, X: pd.DataFrame, max_samples: int = 200):
    """
    Full explainability pipeline:
      - compute SHAP values
      - return explainer, shap_values, X_sample
      - generate summary + single-instance plots
    """
    explainer, shap_values, X_sample = compute_shap_values(model, X, max_samples)

    summary_fig = shap_summary_plot(shap_values, X_sample)
    waterfall_fig = shap_waterfall_plot(shap_values, X_sample, index=0)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "sample": X_sample,
        "summary_fig": summary_fig,
        "waterfall_fig": waterfall_fig,
    }
