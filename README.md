DataKit Pro

DataKit Pro is a no-code, Streamlit-based machine learning toolkit that allows users to upload their own dataset and run a complete ML workflow without writing a single line of code.
Upload a CSV → explore it → preprocess it → select features → train models → evaluate them → view SHAP explainability → export cleaned data and saved models.

🚀 Features

Upload any CSV dataset

Data profiling & EDA

Missing value summary

Descriptive statistics

Correlation heatmap

Preprocessing

Missing value imputation

Outlier removal (IQR / Z-score)

Encoding (One-hot / Ordinal)

Scaling (Standard / MinMax)

Feature Selection (SelectKBest, RFE)

Model Training

Logistic Regression

Decision Tree

Random Forest

Train/test split + optional cross-validation

Model Evaluation

Accuracy, F1-score, Confusion Matrix

Regression metrics & plots

Explainability (SHAP)

Global feature importance

Local explanations

Export Tools

Cleaned dataset

Saved model pipeline

Metrics and report files

📁 Project Structure
datakit-pro/
├─ app.py
├─ ds_toolkit/
│  ├─ preprocessing.py
│  ├─ feature_select.py
│  ├─ models.py
│  ├─ explain.py
│  └─ utils.py
├─ tests/
│  ├─ test_preprocessing.py
│  └─ test_models.py
├─ docs/
│  └─ architecture.md
├─ .vscode/
│  ├─ settings.json
│  ├─ launch.json
│  └─ tasks.json
├─ requirements.txt
├─ .gitignore
└─ README.md

▶️ How to Run

Create a virtual environment:

python -m venv .venv


Activate it:

Windows: .venv\Scripts\Activate.ps1

Mac/Linux: source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

🛠 Tech Stack

Python

Streamlit

Pandas, NumPy

Scikit-learn

SHAP

Matplotlib, Seaborn

📌 License

Open-source. Free to use and extend.
