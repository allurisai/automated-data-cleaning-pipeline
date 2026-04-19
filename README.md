# Automated Data Cleaning Pipeline for ML Applications

An end-to-end automated data cleaning system that detects and resolves common data quality issues — missing values, duplicates, outliers, skewness, and class imbalance — with measurable impact on downstream machine learning model performance.

Built as part of an independent research project at the **University of Utah, School of Computing** under **Prof. El Kindi Rezig**.

---

## Features

- **Issue Detection Engine** — Automatically scans datasets for missing values, duplicate rows, skewness, outliers (IQR method), class imbalance, and feature correlations
- **6+ Cleaning Operations** — Handle missing values (drop/mean/median/zero), remove duplicates, remove outliers, correct skewness (log/sqrt/cbrt), balance classes (RandomOverSampler/RandomUnderSampler), and basic text cleaning
- **Performance Evaluation** — Trains and compares ML models (Linear Regression, Decision Tree, Random Forest, Logistic Regression, Naive Bayes) before and after cleaning to quantify improvement
- **Dual Processing** — Loads datasets using both Pandas and PySpark, comparing processing speeds
- **Interactive UI** — Streamlit-based web interface for uploading data, visualizing issues, applying cleaning steps, and downloading cleaned datasets
- **Dataset Independent** — Works with any CSV dataset across regression, classification, and text classification tasks

---

## Results

| Task | Metric | Before Cleaning | After Cleaning |
|------|--------|----------------|---------------|
| Regression (House Prices) | RMSE | 150,000+ | 0.215 |
| Classification (Titanic) | Accuracy | 82.3% | 90.4% |
| Text Classification (Tweets) | Accuracy | 73.3% | 88.1% |

---

## Tech Stack

- **Python** — Core language
- **PySpark** — Distributed data processing
- **Pandas / NumPy** — Data manipulation
- **Streamlit** — Web application UI
- **scikit-learn** — ML model training and evaluation
- **imbalanced-learn** — Class balancing (RandomOverSampler, RandomUnderSampler)
- **Matplotlib / Seaborn** — Data visualization
- **ydata-profiling** — Automated dataset profiling

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/allurisai/automated-data-cleaning-pipeline.git
cd automated-data-cleaning-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## System Architecture
