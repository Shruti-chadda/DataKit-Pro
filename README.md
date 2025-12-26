# ⚙️ DataKit Pro – No-Code Machine Learning Toolkit

DataKit Pro is a **no-code, interactive Machine Learning application** built using **Streamlit** that allows users to upload any CSV dataset and perform the complete ML workflow **without writing a single line of code**.

This project is designed for **students, beginners, data analysts**, and is scalable for future enhancements.

---

## 🚀 Features

- 📂 Upload any CSV dataset
- 🧹 Automatic data preprocessing
  - Missing value handling (mean, median, mode, KNN)
  - Outlier handling (IQR, Z-Score)
  - Encoding (One-Hot, Ordinal)
  - Scaling (Standard, Min-Max)
- 🧬 Feature selection
  - K-Best
  - Recursive Feature Elimination (RFE)
  - Correlation-based selection
- 🤖 Supports **both Classification & Regression**
- 🧠 Automatic problem-type detection
- 📊 Multiple ML models

### Classification Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Regression Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

- 📈 Model evaluation
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - MSE, RMSE, MAE, R²
- 🔍 Model Explainability using **SHAP**
  - Feature importance (summary plot)
  - Single prediction explanation (waterfall plot)
- 📦 Export full report (metrics + plots + cleaned dataset) as ZIP

---

## 🧠 How It Works

1. Upload a CSV file
2. Select the target column
3. The app automatically detects:
   - **Classification** (categorical / few unique values)
   - **Regression** (continuous numeric target)
4. Apply preprocessing options
5. Select features
6. Train machine learning models
7. View metrics, plots, and SHAP explanations
8. Download a complete report

---

DataKit-Pro/
│
├── app.py # Streamlit application
├── requirements.txt # Dependencies
├── README.md # Documentation
├── .gitignore
│
├── ds_toolkit/ # Core ML logic
│ ├── preprocessing.py
│ ├── feature_select.py
│ ├── models.py
│ ├── explain.py
│ └── utils.py
│
├── docs/
│ └── architecture.md # System design overview

## ▶️ Run the App Locally

### 1️⃣ Clone the repository

git clone https://github.com/Shruti-chadda/DataKit-Pro.git
cd DataKit-Pro

2️⃣ Create a virtual environment
python -m venv .venv

3️⃣ Activate it (Windows)
.venv\Scripts\activate

4️⃣ Install dependencies
pip install -r requirements.txt

5️⃣ Run the app
streamlit run app.py
Open the browser link shown in the terminal.

📦 Future Enhancements
🔸 Automated EDA report

🔸 Hyperparameter tuning

🔸 Model comparison dashboard

🔸 Save & load trained models

🔸 Deployment pipelines

👩‍💻 Author
Shruti Chadda
Data Science & Machine Learning Enthusiast

⭐ Support
If you like this project, please ⭐ star the repository on GitHub!
