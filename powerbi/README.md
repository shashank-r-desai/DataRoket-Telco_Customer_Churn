# Power BI Directory

This folder contains the Power BI reporting assets used to present executive-level customer retention insights.

## Files
- **`Churn Dashboard.pbix`**: Interactive Power BI report file built on top of the cleaned Telco churn data.

## Dashboard Highlights
- **Executive KPIs**: Total customers, churned count, overall churn rate (26.54%), and average account tenure.
- **Contract & Billing Analysis**: Visual churn rate comparison showing Month-to-month contracts (~42% churn) versus multi-year commitments (~3%).
- **Payment Method Impact**: Highlighting payment friction associated with Electronic Checks.
- **Tenure Progression**: Churn curves showing customer attrition risk decreasing steadily as customer tenure lengthens.
- **Scatter Spend Distributions**: Monthly charges plotted against tenure across retained and churned cohorts.

## Complementary Web App
While this Power BI dashboard provides macro-level retrospective reporting, real-time single-customer risk scoring and 3D segmentation are deployed live in our Streamlit web application at:  
👉 [https://telco-customerchurn.streamlit.app/](https://telco-customerchurn.streamlit.app/)
