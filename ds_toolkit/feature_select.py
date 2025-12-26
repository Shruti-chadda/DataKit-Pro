import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ---------------------------------------------------------
# Helper: Auto-detect problem type from target column
# ---------------------------------------------------------
import pandas as pd

def detect_problem_type(y: pd.Series):
    """Return 'classification' or 'regression' based on target column datatype."""
    # If numeric dtype -> regression, else classification
    if pd.api.types.is_numeric_dtype(y):
        return "regression"
    return "classification"



# ---------------------------------------------------------
# SelectKBest Feature Selection
# ---------------------------------------------------------
def select_kbest(X, y, k=10):
    """Apply SelectKBest using ANOVA F-test (classification) or F-regression."""
    problem = detect_problem_type(y)

    if problem == "classification":
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    else:
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))

    selector.fit(X, y)

    selected_cols = X.columns[selector.get_support()].tolist()
    scores = selector.scores_

    return X[selected_cols], selected_cols, scores


# ---------------------------------------------------------
# Recursive Feature Elimination (RFE)
# ---------------------------------------------------------
def select_rfe(X, y, n_features=10):
    """Apply RFE using a simple RandomForest estimator."""
    problem = detect_problem_type(y)

    if problem == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    selector = RFE(model, n_features_to_select=min(n_features, X.shape[1]))
    selector.fit(X, y)

    selected_cols = X.columns[selector.support_].tolist()
    rankings = selector.ranking_

    return X[selected_cols], selected_cols, rankings


# ---------------------------------------------------------
# Correlation-Based Feature Selection
# ---------------------------------------------------------
def select_correlated_features(df: pd.DataFrame, target_col: str, threshold=0.1):
    """
    Select features based on correlation with target variable.
    - threshold: minimum |correlation| needed to keep a feature.
    """
    corr = df.corr(numeric_only=True)
    target_corr = corr[target_col].abs().sort_values(ascending=False)

    # Select features above threshold (excluding the target itself)
    selected = target_corr[target_corr >= threshold].index.tolist()
    if target_col in selected:
        selected.remove(target_col)

    return selected, target_corr


# ---------------------------------------------------------
# Unified Feature Selection Interface
# ---------------------------------------------------------
def apply_feature_selection(X, y, method="none", k=10):
    """
    Apply different feature selection methods.
    method: "none", "kbest", "rfe", "corr"
    """
    if method == "none":
        return X, X.columns.tolist()

    if method == "kbest":
        X_new, cols, _ = select_kbest(X, y, k=k)
        return X_new, cols

    if method == "rfe":
        X_new, cols, _ = select_rfe(X, y, n_features=k)
        return X_new, cols

    if method == "corr":
        df = pd.concat([X, y], axis=1)
        selected, _ = select_correlated_features(df, y.name)
        return X[selected], selected

    raise ValueError("Invalid feature selection method.")
