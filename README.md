# ⚙️ DataKit Pro – Machine Learning Toolkit

DataKit Pro is a **no-code, interactive machine learning tool** built with Streamlit.
Upload any CSV dataset → preprocess → select features → train ML models → evaluate performance — all without writing a single line of code.

This project is designed for **students, beginners, data analysts, and future scalability**, making ML workflows fast, visual, and easy to understand.

---

## 🚀 Features

✔ Upload any CSV dataset
✔ Automatic data cleaning (missing values, encoding, scaling)
✔ Feature selection (K-Best, Variance Threshold)
✔ Supports **both classification & regression**
✔ Multiple ML models:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
  ✔ Automatic problem-type detection (classification vs regression)
  ✔ Evaluation metrics:
* Accuracy
* Confusion Matrix
* MSE, MAE, R² for regression
  ✔ Simple, user-friendly Streamlit interface

---

## 📂 Project Structure

```
DataKit-Pro/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Optional project config
├── README.md              # Documentation
├── .gitignore             # Ignored files
│
├── ds_toolkit/            # Core ML logic
│   ├── preprocessing.py
│   ├── feature_select.py
│   ├── models.py
│   ├── explain.py
│   └── utils.py
│
└── docs/
    └── architecture.md    # Project design overview
```

---

## 🧠 How It Works

1️⃣ Upload a CSV file
2️⃣ Choose the target column
3️⃣ The app automatically detects whether it’s:

* **Classification** (categorical target)
* **Regression** (numeric target)

4️⃣ Choose preprocessing options
5️⃣ Select ML models
6️⃣ Train & evaluate
7️⃣ View accuracy, metrics, and graphs

---

## 🧪 Supported Machine Learning Models

### Classification

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### Regression

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

---

## ▶️ Running the App Locally

### 1. Create a virtual environment

```
python -m venv .venv
```

### 2. Activate it

**Windows PowerShell:**

```
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the app

```
streamlit run app.py
```

---

## 📦 Future Enhancements

🔸 Model explainability (SHAP, LIME)
🔸 Automated EDA report
🔸 Model comparison dashboard
🔸 Save & load trained models
🔸 Hyperparameter tuning

---

## 👩‍💻 Author

**Shruti Chadda**
Data Science & Machine Learning Enthusiast

---

## ⭐ Show Your Support

If you like this project, please ⭐ **star this repository** on GitHub!

---

