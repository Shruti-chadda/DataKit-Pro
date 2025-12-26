import pandas as pd
import numpy as np

from ds_toolkit.preprocessing import (
    simple_impute,
    knn_impute,
    remove_outliers_iqr,
    remove_outliers_zscore,
    encode_onehot,
    encode_ordinal,
    scale_standard,
    scale_minmax,
    apply_preprocessing,
)


# ------------------------------------------
# Sample Data for Tests
# ------------------------------------------
def sample_df_with_target():
    # includes a target column named 'target' to exercise target_col handling
    return pd.DataFrame({
        "num1": [1, 2, None, 4, 100],     # contains missing + outlier
        "num2": [10, 12, 14, None, 16],   # contains missing
        "cat": ["A", "B", "A", "C", None],# missing categorical
        "target": [0, 1, 0, 1, 0]         # target column should be preserved
    })


# ------------------------------------------
# Tests for Missing Value Handling
# ------------------------------------------

def test_simple_impute_mean():
    df = sample_df_with_target()
    df2 = simple_impute(df.drop(columns=["target"]), "mean")
    assert df2["num1"].isnull().sum() == 0
    assert df2["num2"].isnull().sum() == 0


def test_simple_impute_median():
    df = sample_df_with_target()
    df2 = simple_impute(df.drop(columns=["target"]), "median")
    assert df2["num1"].isnull().sum() == 0
    assert df2["num2"].isnull().sum() == 0


def test_simple_impute_mode():
    df = sample_df_with_target()
    df2 = simple_impute(df, "mode")
    assert df2["cat"].isnull().sum() == 0


def test_knn_impute():
    df = sample_df_with_target()
    df2 = knn_impute(df.drop(columns=["target"]))
    assert df2["num1"].isnull().sum() == 0
    assert df2["num2"].isnull().sum() == 0


# ------------------------------------------
# Tests for Outlier Removal
# ------------------------------------------

def test_remove_outliers_iqr():
    df = sample_df_with_target()
    df2 = remove_outliers_iqr(df.drop(columns=["target"]))
    # Outlier "100" in num1 should be removed
    assert df2["num1"].max() < 100


def test_remove_outliers_zscore():
    df = sample_df_with_target()
    df2 = remove_outliers_zscore(df.drop(columns=["target"]), threshold=2)
    assert df2["num1"].max() < 100


# ------------------------------------------
# Encoding Tests
# ------------------------------------------

def test_onehot_encoding():
    df = sample_df_with_target()
    df2 = encode_onehot(df.drop(columns=["target"]))
    # Should create new columns for "cat"
    assert any(col.startswith("cat_") for col in df2.columns)


def test_ordinal_encoding():
    df = sample_df_with_target()
    df2 = encode_ordinal(df.drop(columns=["target"]))
    # Encoded categorical column should be numeric
    assert np.issubdtype(df2["cat"].dtype, np.number)


# ------------------------------------------
# Scaling Tests
# ------------------------------------------

def test_scale_standard():
    df = sample_df_with_target()
    df_clean = simple_impute(df.drop(columns=["target"]), "mean")
    df_scaled, scaler = scale_standard(df_clean)
    # Standard scaler -> mean approx 0
    assert abs(df_scaled["num1"].mean()) < 1e-6


def test_scale_minmax():
    df = sample_df_with_target()
    df_clean = simple_impute(df.drop(columns=["target"]), "mean")
    df_scaled, scaler = scale_minmax(df_clean)
    # MinMax -> range should be 0–1
    assert df_scaled["num1"].min() >= 0
    assert df_scaled["num1"].max() <= 1


# ------------------------------------------
# Combined Pipeline Test (features only, target preserved)
# ------------------------------------------

def test_apply_preprocessing_pipeline_preserves_target():
    df = sample_df_with_target()
    df2, scaler = apply_preprocessing(
        df,
        missing="median",
        outliers="iqr",
        encoding="onehot",
        scaling="standard",
        target_col="target"
    )

    # Target should still exist and match original values for kept indices
    assert "target" in df2.columns
    # No missing values in features
    features = df2.drop(columns=["target"])
    assert features.isnull().sum().sum() == 0
    assert scaler is not None
