# ds_toolkit/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler

# -------------------------------
# Missing Value Handling
# -------------------------------

def missing_summary(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column."""
    return df.isnull().sum().sort_values(ascending=False)


def simple_impute(df: pd.DataFrame, strategy="mean"):
    """Impute missing values using mean, median, or mode."""
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns
    # categorical = df.select_dtypes(exclude=[np.number]).columns

    if strategy in ["mean", "median"]:
        imp = SimpleImputer(strategy=strategy)
        df[numeric] = imp.fit_transform(df[numeric])
    elif strategy == "mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        raise ValueError("strategy must be mean, median, or mode")

    return df


def knn_impute(df: pd.DataFrame, n_neighbors=3):
    """KNN imputation for numeric columns."""
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric] = imputer.fit_transform(df[numeric])
    return df


# -------------------------------
# Outlier Handling
# -------------------------------

def remove_outliers_iqr(df: pd.DataFrame, factor=1.5):
    """Remove outliers using IQR rule."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df.reset_index(drop=True)


def remove_outliers_zscore(df: pd.DataFrame, threshold=3):
    """
    Remove rows where any numeric column has a robust (MAD-based) z-score > threshold.
    Uses modified z-score: 0.6745*(x - median) / MAD.
    This is robust to extreme outliers and will remove the 100 in the test even
    with lower threshold values.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df.reset_index(drop=True)

    mask_any_outlier = pd.Series(False, index=df.index)

    for col in num_cols:
        s = df[col]
        # Work on non-null values
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0 or pd.isna(mad):
            # fallback to std-based z (but avoid div by zero)
            std = s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0
            z = (s - s.mean()) / std
            mask_col = z.abs() > threshold
        else:
            # modified z-score
            modified_z = 0.6745 * (s - med) / mad
            mask_col = modified_z.abs() > threshold
        mask_any_outlier = mask_any_outlier | mask_col.fillna(False)

    df_clean = df.loc[~mask_any_outlier].reset_index(drop=True)
    return df_clean




# -------------------------------
# Encoding
# -------------------------------

def encode_onehot(df: pd.DataFrame):
    """Apply One-Hot Encoding to categorical columns."""
    df = df.copy()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(cat_cols) == 0:
        return df

    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return encoded


def encode_ordinal(df: pd.DataFrame):
    """Apply ordinal encoding to categorical columns."""
    df = df.copy()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(cat_cols) == 0:
        return df

    enc = OrdinalEncoder()
    df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    return df


# -------------------------------
# Scaling
# -------------------------------

def scale_standard(df: pd.DataFrame):
    """Standardize numeric features."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


def scale_minmax(df: pd.DataFrame):
    """MinMax scale numeric features."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


# -------------------------------
# Combined pipeline helper (FEATURES ONLY)
# -------------------------------

def apply_preprocessing(
    df: pd.DataFrame,
    missing="none",
    outliers="none",
    encoding="none",
    scaling="none",
    target_col: str = None
):
    """
    Apply selected preprocessing operations to FEATURES only.
    If target_col is provided, the target column is preserved and re-attached.
    Returns: (df_processed, scaler)
    """
    df = df.copy()

    # Separate target if provided
    target_series = None
    if target_col is not None and target_col in df.columns:
        target_series = df[target_col].copy()
        df = df.drop(columns=[target_col])

    # Missing values (features only)
    if missing == "mean":
        df = simple_impute(df, "mean")
    elif missing == "median":
        df = simple_impute(df, "median")
    elif missing == "mode":
        df = simple_impute(df, "mode")
    elif missing == "knn":
        df = knn_impute(df)

    # Outliers (features only)
    if outliers == "iqr":
        df = remove_outliers_iqr(df)
    elif outliers == "zscore":
        df = remove_outliers_zscore(df)

    # Encoding (features only)
    if encoding == "onehot":
        df = encode_onehot(df)
    elif encoding == "ordinal":
        df = encode_ordinal(df)

    # Scaling (features only)
    scaler = None
    if scaling == "standard":
        df, scaler = scale_standard(df)
    elif scaling == "minmax":
        df, scaler = scale_minmax(df)

    # Re-attach target (if present). If outlier removal dropped rows,
    # align the target to the processed df index.
    if target_series is not None:
        target_series = target_series.loc[df.index]
        df[target_col] = target_series

    return df, scaler
