import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(page_title="DataKit Pro", layout="wide")
st.title("⚙️ DataKit Pro – Adaptive ML Toolkit")


# -----------------------------------------------------------
# 1. Upload dataset
# -----------------------------------------------------------
uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded is None:
    st.info("Please upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# -----------------------------------------------------------
# 2. Target Selection (Adaptive)
# -----------------------------------------------------------
st.subheader("🎯 Target Selection")

target_col = st.selectbox("Select target column", df.columns)
y = df[target_col]
X = df.drop(columns=[target_col])

# Transform categorical target into numeric automatically
y_original = y.copy()

if y.dtype == "object" or y.dtype.name == "category":
    le = LabelEncoder()
    y = le.fit_transform(y)
    st.success(f"Automatically encoded target classes: {list(le.classes_)}")

# Check if target is regression or classification
unique_vals = np.unique(y)

if len(unique_vals) <= 10 and set(unique_vals).issubset(set(range(10))):
    problem_type = "classification"
else:
    problem_type = "regression"

st.write(f"🧠 **Detected Problem Type:** {problem_type.upper()}")


# -----------------------------------------------------------
# 3. Preprocessing
# -----------------------------------------------------------
st.subheader("🧹 Preprocessing (Automatic)")

X_proc = X.copy()

# Fill numeric missing values
num_cols = X_proc.select_dtypes(include=[np.number]).columns
for c in num_cols:
    X_proc[c] = X_proc[c].fillna(X_proc[c].median())

# Fill categorical missing values
cat_cols = X_proc.select_dtypes(exclude=[np.number]).columns
for c in cat_cols:
    X_proc[c] = X_proc[c].fillna(X_proc[c].mode().iloc[0])

# One-hot encode categoricals
X_proc = pd.get_dummies(X_proc, drop_first=True)

# Scale numeric features
scaler = StandardScaler()
num_cols_proc = X_proc.select_dtypes(include=[np.number]).columns
X_proc[num_cols_proc] = scaler.fit_transform(X_proc[num_cols_proc])

st.write("Processed dataset:")
st.dataframe(X_proc.head())


# -----------------------------------------------------------
# 4. Train/Test Split
# -----------------------------------------------------------
st.subheader("📊 Train / Test Split")
test_size_pct = st.slider("Test size (%)", 10, 40, 20)
test_size = test_size_pct / 100

X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y, test_size=test_size, random_state=42
)

st.write(f"Training rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")


# -----------------------------------------------------------
# 5. Model Selection (Adaptive)
# -----------------------------------------------------------
st.subheader("🤖 Model Training")

if problem_type == "classification":
    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200),
        "SVM Classifier": SVC(),
    }
else:
    model_options = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200),
        "Decision Tree Regressor": DecisionTreeRegressor(),
    }

models_selected = st.multiselect(
    "Choose models to train:",
    list(model_options.keys()),
    default=list(model_options.keys())[:2]
)

if st.button("Train Models"):
    results = {}

    for name in models_selected:
        model = model_options[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {"MSE": mse, "R²": r2}

    st.success("Training complete!")

    st.write("### 📈 Results")
    st.json(results)

    # Confusion matrix for classification only
    if problem_type == "classification":
        st.write("### 🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)


# -----------------------------------------------------------
# 6. Export (future scope placeholder)
# -----------------------------------------------------------
st.subheader("📦 Export Options (Future Scope)")
st.info("Reports, model export, and pipelines will be added in future updates.")
