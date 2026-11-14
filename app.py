import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="Telco Customer Churn & Retention AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODEL & DATA CACHING
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    models_path = os.path.join(BASE_DIR, "models")
    pipeline = joblib.load(os.path.join(models_path, "churn_pipeline.joblib"))
    clustering = joblib.load(os.path.join(models_path, "customer_clustering.joblib"))
    meta = joblib.load(os.path.join(models_path, "feature_metadata.joblib"))
    return pipeline, clustering, meta

@st.cache_data
def load_dataset():
    data_path = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df

try:
    pipeline, clustering, meta = load_artifacts()
    df_raw = load_dataset()
except Exception as e:
    st.error(f"Error loading models or dataset: {e}")
    st.stop()

# ==============================================================================
# SIDEBAR NAVIGATION & APP HEADER
# ==============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
st.sidebar.title("Telco Retention AI")
st.sidebar.caption("Machine Learning & Customer Segmentation Engine")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🎯 1. Live Churn Risk Predictor",
        "🧩 2. Customer Segmentation Explorer",
        "📈 3. Model Explainability & Drivers",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Specs:**
- **Algorithm:** Random Forest Classifier
- **Ensemble Trees:** 150 Estimators
- **Class Balance:** Balanced Weights
- **ROC-AUC Score:** 0.8426
- **Clustering:** K-Means ($k=3$)
""")

# ==============================================================================
# 1. LIVE CHURN RISK PREDICTOR
# ==============================================================================
if nav_choice == "🎯 1. Live Churn Risk Predictor":
    st.title("🎯 Live Customer Churn Risk Predictor")
    st.markdown("Enter customer account details below to predict churn likelihood and retrieve targeted retention actions.")
    
    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            
            st.subheader("📞 Phone Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multilines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            
        with col2:
            st.subheader("🌐 Internet & Add-ons")
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
        with col3:
            st.subheader("💳 Account & Billing")
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=0.5)
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(tenure * monthly), step=10.0)

        submit_btn = st.form_submit_button("⚡ Evaluate Churn Risk", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multilines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])
        
        churn_prob = pipeline.predict_proba(input_data)[0][1]
        prediction = 1 if churn_prob >= 0.50 else 0
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.subheader("Prediction Outcome")
            if churn_prob >= 0.50:
                st.error(f"🚨 **HIGH RISK OF CHURNING** (Probability: {churn_prob*100:.1f}%)")
            else:
                st.success(f"✅ **LOW RISK / LOYAL CUSTOMER** (Probability: {churn_prob*100:.1f}%)")
                
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#D32F2F" if churn_prob >= 0.5 else "#388E3C"},
                    'steps': [
                        {'range': [0, 40], 'color': "#E8F5E9"},
                        {'range': [40, 70], 'color': "#FFF9C4"},
                        {'range': [70, 100], 'color': "#FFEBEE"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.subheader("Actionable Retention Strategy")
            if churn_prob >= 0.50:
                st.markdown("""
                **Identified Risk Drivers & Recommended Interventions:**
                - 📉 **Contract Vulnerability**: If on *Month-to-Month*, offer an annual incentive discount of 15%.
                - 💳 **Billing Friction**: Switch from *Electronic Check* to *Automated Bank/Card* with a one-time bill credit.
                - 🛠️ **Service Deficit**: Bundle *TechSupport* or *OnlineSecurity* complimentary for 3 months.
                - 📞 **Agent Action**: Trigger an immediate proactive call from customer success within 48 hours.
                """)
            else:
                st.markdown("""
                **Upsell & Advocacy Plan:**
                - ⭐ **High Loyalty Score**: Customer is stable.
                - 🚀 **Expansion Opportunity**: Recommend speed upgrades or family add-on streaming bundles.
                - 🎁 **Loyalty Reward**: Send a loyalty appreciation anniversary voucher.
                """)

# ==============================================================================
# 2. CUSTOMER SEGMENTATION EXPLORER
# ==============================================================================
elif nav_choice == "🧩 2. Customer Segmentation Explorer":
    st.title("🧩 Unsupervised Customer Segmentation (K-Means)")
    st.markdown("K-Means clustering partitions customers into 3 behavioral segments based on **Tenure**, **Monthly Charges**, and **Total Lifetime Spend**.")
    
    kmeans_model = clustering["kmeans"]
    scaler = clustering["scaler"]
    
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_scaled = scaler.transform(df_raw[num_cols])
    df_raw["Cluster"] = kmeans_model.predict(X_scaled)
    
    # Assign human-readable cluster personas
    cluster_names = {
        0: "Segment 0: New / Budget Shoppers",
        1: "Segment 1: High-Spend / Long-Term Loyalists",
        2: "Segment 2: Moderate Tenure / Tech Streamers"
    }
    df_raw["Segment_Name"] = df_raw["Cluster"].map(cluster_names)
    
    col1, col2, col3 = st.columns(3)
    for c_id, col in zip([0, 1, 2], [col1, col2, col3]):
        c_df = df_raw[df_raw["Cluster"] == c_id]
        churn_rate = (c_df["Churn"] == "Yes").mean() * 100
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{cluster_names[c_id]}</h4>
                <p><b>Size:</b> {len(c_df):,} customers</p>
                <p><b>Avg Tenure:</b> {c_df['tenure'].mean():.1f} months</p>
                <p><b>Avg Monthly Spend:</b> ${c_df['MonthlyCharges'].mean():.2f}</p>
                <p><b>Churn Rate:</b> <span class="{'badge-high' if churn_rate > 30 else 'badge-low'}">{churn_rate:.1f}%</span></p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader("3D Interactive Cluster Visualization")
    
    sample_df = df_raw.sample(n=min(1500, len(df_raw)), random_state=42)
    fig_3d = px.scatter_3d(
        sample_df,
        x="tenure",
        y="MonthlyCharges",
        z="TotalCharges",
        color="Segment_Name",
        symbol="Churn",
        opacity=0.7,
        title="Customer Persona Landscape (Sample of 1,500 Customers)",
        labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly ($)", "TotalCharges": "Total Spend ($)"}
    )
    fig_3d.update_layout(height=650, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_3d, use_container_width=True)

# ==============================================================================
# 3. MODEL EXPLAINABILITY & DRIVERS
# ==============================================================================
elif nav_choice == "📈 3. Model Explainability & Drivers":
    st.title("📈 Model Evaluation & Key Churn Drivers")
    st.markdown("Inspect performance metrics and identify the most significant factors causing customer attrition.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC Score", "0.8426", "+0.04 vs Decision Tree")
    col2.metric("Recall (Churners)", "79.1%", "Prioritized for Retention")
    col3.metric("F1-Score", "0.631", "Balanced Trade-off")
    col4.metric("Total Test Accuracy", "75.2%", "Evaluated on 1,409 customers")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Top Feature Importances (Random Forest)")
        classifier = pipeline.named_steps["classifier"]
        preproc = pipeline.named_steps["preprocessor"]
        
        cat_features = meta["categorical_features"]
        num_features = meta["numeric_features"]
        cat_encoded = preproc.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_features)
        feature_names = num_features + list(cat_encoded)
        
        importances = classifier.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(12)
        
        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Top 12 Drivers Influencing Churn Prediction"
        )
        fig_feat.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with col_right:
        st.subheader("Churn Rate by Key Drivers")
        driver_tab = st.selectbox("Select Dimension to Inspect:", ["Contract", "InternetService", "PaymentMethod", "TechSupport"])
        
        driver_analysis = df_raw.groupby(driver_tab)["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        driver_analysis.columns = [driver_tab, "Churn Rate (%)"]
        
        fig_driver = px.bar(
            driver_analysis,
            x=driver_tab,
            y="Churn Rate (%)",
            color="Churn Rate (%)",
            color_continuous_scale="Reds",
            title=f"Actual Churn Rate by {driver_tab}"
        )
        fig_driver.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_driver, use_container_width=True)
