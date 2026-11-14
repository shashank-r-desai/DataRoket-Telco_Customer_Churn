# Telco Customer Churn & Retention AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-customerchurn.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning, Customer Segmentation, and Business Analytics solution to predict and reduce subscriber churn in the telecommunications sector.

🔗 **Live Public Application:** [https://telco-customerchurn.streamlit.app/](https://telco-customerchurn.streamlit.app/)

---

## 📌 Project Overview
Customer acquisition costs 5× to 7× more than customer retention. This project combines **Supervised Machine Learning** (Random Forest classification), **Unsupervised Machine Learning** (K-Means persona segmentation), and **Business Intelligence** (Power BI & Streamlit) to identify at-risk customers *before* they leave and deliver tailored retention offers.

### Core Objectives
1. **Zero Data Leakage Pipeline**: Strict Scikit-Learn `ColumnTransformer` + `Pipeline` preventing lookahead contamination.
2. **Supervised Classification**: Train Decision Tree and Random Forest classifiers; optimize for **Recall** (catching ~80% of churners) and **ROC-AUC (0.8426)**.
3. **Unsupervised Customer Personas**: Segment subscribers using K-Means ($k=3$) across tenure, monthly charges, and lifetime spend.
4. **Interactive Cloud App**: Deploy a live multi-page Streamlit web app for real-time predictions and 3D segmentation exploration.
5. **Business Intelligence**: Power BI dashboard for organizational reporting and high-level KPIs.

---

## 🚀 Live Streamlit Web App (`app.py`)
Access the live deployed web app: **[telco-customerchurn.streamlit.app](https://telco-customerchurn.streamlit.app/)**

### Application Features:
- **🎯 1. Live Churn Risk Predictor**: Enter customer demographics, services, and billing details to receive instant churn probabilities, risk levels, and targeted retention offers.
- **🧩 2. Customer Segmentation Explorer**: Explore 3 distinct customer personas with KPI summaries and an interactive 3D scatter plot.
- **📈 3. Model Explainability & Drivers**: Inspect top feature importances (Contract type, Tenure, Electronic checks) and dimensional churn drill-downs.

---

## 📊 Model Performance & Evaluation

The dataset is imbalanced (~73% retained vs ~27% churned). Because missing a churning customer (False Negative) loses permanent Customer Lifetime Value (LTV), models were optimized for **Recall** and **ROC-AUC** using `class_weight='balanced'`.

| Model | ROC-AUC | Recall (Churn) | Precision | F1-Score | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Pruned)** | 0.8012 | 74.20% | 48.50% | 0.5860 | 73.80% |
| **Random Forest (Production)** | **0.8426** | **79.14%** | **52.10%** | **0.6310** | **75.23%** |

*Evaluated on an isolated test set of 1,409 unseen customer accounts.*

---

## 📂 Repository Structure
```
FUTURE_DS_02/
│
├── data/
│   ├── Telco-Customer-Churn.csv         # Raw IBM Telco dataset (~7,043 rows)
│   ├── cleaned_churn_data.csv           # Cleaned dataset exported
│   └── README.md
│
├── models/
│   ├── churn_pipeline.joblib            # End-to-end Scikit-Learn pipeline (transformers + Random Forest)
│   ├── customer_clustering.joblib       # K-Means clustering model + numeric scaler
│   ├── feature_metadata.joblib          # Feature schema & validation metadata
│   └── README.md
│
├── notebooks/
│   ├── Telco_Customer_Churn.ipynb       # Complete notebook (EDA, Preprocessing, ML, Clustering, Persistence)
│   └── README.md
│
├── powerbi/
│   ├── Churn Dashboard.pbix             # Interactive Power BI report file
│   └── README.md
│
├── visuals/
│   ├── Churn Dashboard.png              # Power BI report screenshot
│   └── README.md
│
├── app.py                               # Production Streamlit Cloud Web Application
├── requirements.txt                     # Dependencies for deployment
├── Viva_and_Presentation_Guide.docx     # Comprehensive presentation & viva defense guide
├── LICENSE
└── README.md
```

---

## 🛠️ Installation & Local Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shashank-r-desai/FUTURE_DS_02.git
   cd FUTURE_DS_02
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app locally:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## 💡 Key Business Insights
- **Contract Type**: Month-to-month contracts exhibit an alarming **~42% churn rate**, compared to just **~3%** for two-year contracts.
- **Tenure Vulnerability**: Customers in their first 6 months have the highest risk of attrition; retention programs must focus on onboarding.
- **Payment Method**: Subscribers paying via **Electronic Check** churn at nearly double the rate of those on automated bank or credit card transfers.
- **Value-Added Support**: Lack of **Tech Support** and **Online Security** strongly correlates with increased churn risk.
