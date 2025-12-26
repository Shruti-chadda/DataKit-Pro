import pandas as pd
from ds_toolkit.models import train_models, detect_problem_type_from_y


# ------------------------------------------
# Sample synthetic dataset for testing
# ------------------------------------------
def sample_classification_data():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 50],
        "salary": [30, 35, 40, 42, 50, 60],
        "label": [0, 1, 0, 1, 0, 1]
    })
    X = df[["age", "salary"]]
    y = df["label"]
    return X, y


def sample_regression_data():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": [2, 4, 6, 8, 10, 12],
        "target": [3, 6, 7, 8, 13, 14]
    })
    X = df[["feature1", "feature2"]]
    y = df["target"]
    return X, y


# ------------------------------------------
# Tests
# ------------------------------------------

def test_detect_problem_type_classification():
    _, y = sample_classification_data()
    assert detect_problem_type_from_y(y) == "classification"


def test_detect_problem_type_regression():
    _, y = sample_regression_data()
    assert detect_problem_type_from_y(y) == "regression"


def test_train_models_classification():
    X, y = sample_classification_data()

    results = train_models(
        X, y,
        models_to_run=["logistic_regression"],
        test_size=0.3,
        use_cv=False
    )

    assert "logistic_regression" in results["models"]
    assert "accuracy" in results["metrics"]["logistic_regression"]


def test_train_models_regression():
    X, y = sample_regression_data()

    results = train_models(
        X, y,
        models_to_run=["linear_regression"],
        test_size=0.3,
        use_cv=False
    )

    assert "linear_regression" in results["models"]
    assert "mse" in results["metrics"]["linear_regression"]
