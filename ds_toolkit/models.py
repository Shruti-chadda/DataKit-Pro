"""
ds_toolkit/models.py

Core model training, evaluation, and persistence helpers for DataKit Pro.

Functions:
- detect_problem_type_from_y(y)
- train_models(...)
- evaluate_classification(...)
- evaluate_regression(...)
- save_model(...), load_model(...)
- get_default_models(...)
"""

from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# -------------------------
# Helpers
# -------------------------
def detect_problem_type_from_y(y: pd.Series) -> str:
    """
    Return 'classification' or 'regression' based primarily on dtype.
    Numeric dtype -> regression; otherwise classification.
    """
    if pd.api.types.is_numeric_dtype(y):
        return "regression"
    return "classification"


# -------------------------
# Evaluation helpers
# -------------------------
def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute common classification metrics."""
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    if y_proba is not None:
        # For multiclass, try to compute ROC AUC when possible (ovo/ovr complexity avoided)
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                # binary prob or probability for positive class
                if y_proba.ndim == 1:
                    proba_pos = y_proba
                else:
                    proba_pos = y_proba[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
        except Exception:
            # ignore ROC AUC errors
            metrics["roc_auc"] = None
    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute common regression metrics."""
    metrics = {}
    metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    metrics["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))
    return metrics


# -------------------------
# Default models factory
# -------------------------
def get_default_models(problem: str = "classification") -> Dict[str, Any]:
    """Return a dict of default model name -> unfitted estimator based on problem type."""
    if problem == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=200, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
        }
    else:
        return {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "decision_tree_regressor": DecisionTreeRegressor(random_state=42),
        }


# -------------------------
# Training orchestration
# -------------------------
def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    problem: Optional[str] = None,
    models_to_run: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    use_cv: bool = False,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Train one or more models and return results.

    Returns a dict with:
      - 'models': {name: fitted_estimator}
      - 'metrics': {name: metrics_dict}
      - 'cv_scores': {name: list_of_cv_scores} (if use_cv)
      - 'train_size', 'test_size'
    """
    if problem is None:
        problem = detect_problem_type_from_y(y)

    available_models = get_default_models(problem)
    if models_to_run is None:
        models_to_run = list(available_models.keys())

    # Determine stratify argument safely
    stratify_arg = None
    if problem == "classification":
        # require at least 2 samples per class to stratify
        class_counts = y.value_counts()
        if class_counts.min() >= 2:
            stratify_arg = y
    # Now split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )


    results: Dict[str, Any] = {
        "models": {},
        "metrics": {},
        "cv_scores": {},
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "problem_type": problem,
    }

    for name in models_to_run:
        if name not in available_models:
            # skip unknown model names
            continue
        estimator = available_models[name]

        # Fit
        estimator.fit(X_train, y_train)

        # Predict
        y_pred = estimator.predict(X_test)
        y_proba = None
        try:
            if hasattr(estimator, "predict_proba"):
                y_proba = estimator.predict_proba(X_test)
        except Exception:
            y_proba = None

        # Evaluate
        if problem == "classification":
            metrics = evaluate_classification(y_test, y_pred, y_proba)
        else:
            metrics = evaluate_regression(y_test, y_pred)

        results["models"][name] = estimator
        results["metrics"][name] = metrics

        # Cross-validation (optional)
        if use_cv:
            try:
                # For classification use accuracy by default, regression uses neg_mean_squared_error
                scoring = "accuracy" if problem == "classification" else "neg_mean_squared_error"
                cv_scores = cross_val_score(estimator, X, y, cv=cv_folds, scoring=scoring)
                # For regression neg MSE -> convert to positive MSE
                if scoring.startswith("neg_"):
                    cv_scores = -cv_scores
                results["cv_scores"][name] = cv_scores.tolist()
            except Exception:
                results["cv_scores"][name] = None

    return results


# -------------------------
# Persistence helpers
# -------------------------
def save_model(obj: Any, path: str) -> None:
    """Save a model or pipeline to disk using joblib."""
    joblib.dump(obj, path)


def load_model(path: str) -> Any:
    """Load a joblib-saved model/pipeline."""
    return joblib.load(path)
