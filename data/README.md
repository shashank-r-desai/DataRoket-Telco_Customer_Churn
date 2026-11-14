# Data Directory

This folder contains the datasets used across the exploratory analysis, machine learning pipelines, and Power BI dashboards.

## Files
- **`Telco-Customer-Churn.csv`**: The original raw IBM Telco Customer Churn dataset (~7,043 customer accounts, 21 attributes).
- **`cleaned_churn_data.csv`**: The cleaned dataset generated with imputed `TotalCharges`, tenure groupings, and encoded churn flags, formatted for Power BI ingestion.

## Source
- IBM Telco Customer Churn (Kaggle): [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Preprocessing Standards
- Missing values in `TotalCharges` were handled cleanly.
- In machine learning workflows, all imputation and scaling are fitted strictly on the **Training set** to eliminate **Data Leakage**.
