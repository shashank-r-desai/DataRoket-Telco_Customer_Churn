# Notebooks Directory

This folder contains the complete Jupyter Notebook documenting the analysis and machine learning implementation.

## Primary Notebook
- **`Telco_Customer_Churn.ipynb`**: Complete end-to-end data science notebook.

## Key Sections in the Notebook
1. **Problem Statement & Dataset Overview**: Business objectives and feature definitions.
2. **Data Cleaning & Type Corrections**: Handling whitespace blanks in `TotalCharges` and converting types.
3. **Exploratory Data Analysis (EDA)**: Churn distribution, bivariate analyses against tenure, monthly charges, and contract types.
4. **Scikit-Learn Preprocessing Pipeline**: Stratified train-test splitting (80/20) and `ColumnTransformer` (StandardScaler + OneHotEncoder) ensuring **Zero Data Leakage**.
5. **Supervised Classification**: Training pruned **Decision Tree** baseline and ensemble **Random Forest Classifier** with `class_weight='balanced'`.
6. **Model Evaluation Suite**: Detailed comparison of Accuracy, Precision, Recall (79.1%), F1-Score, and ROC-AUC (0.8426), alongside confusion matrices.
7. **Customer Segmentation**: Unsupervised **K-Means Clustering** ($k=3$) to identify behavioral customer personas.
8. **Production Model Persistence**: Serializing pipelines and models via `joblib` for deployment in the Streamlit web application.
