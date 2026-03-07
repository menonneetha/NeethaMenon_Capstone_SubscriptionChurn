# Subscription Churn: Early-Warning & Retention Strategy

## 1. Project Overview
This project provides an end-to-end analytical solution for a subscription-based service experiencing a spike in cancellations (rising from 1.1% to 7.9% over 8 months). It transitions the business from reactive churn reporting to a proactive retention framework using Machine Learning. 

The strategy focuses on identifying "Quiet Churners"—users whose declining engagement (usage_trend) serves as a 14-day leading indicator of cancellation—allowing for targeted interventions that protect high-value revenue.

## 2. Technical Architecture & Methodology

### Data Engineering (ETL)
The pipeline (`etl_pipeline.py`) processes raw subscription and usage data to generate three curated outputs:
1. **dim_users_enriched.csv**: User demographics merged with behavioral engagement bands.
2. **fact_user_weekly.csv**: Time-series aggregations for Cohort Analysis.
3. **model_churn_dataset.csv**: Finalized feature set for Machine Learning, including the critical `usage_trend` metric.

### Machine Learning & Analysis (`analysis.py`)
The analysis engine utilizes a **Random Forest Classifier** to predict churn probability. 
- **Threshold Optimization:** A business-weighted threshold of **0.3** is applied to prioritize "Recall," ensuring the business captures the maximum number of at-risk users.
- **Model Validation:** Calculations are performed on a **20% holdout test set** to ensure statistical rigour and prevent over-fitting.
- **Automated Reporting:** The script automatically generates diagnostic visualizations (Feature Importance, Confusion Matrix, Usage KDE plots) into the `/dashboard/dashboard_screenshots/` folder.

## 3. Data Reconciliation & Business Case
To ensure transparency across project deliverables, note the following data scopes:
- **Executive Deck Tables:** Represent the **Total Population** (100% of historical data) to show macro-revenue leakage (~$216k total).
- **Python Projections:** Represent the **Validation Set** (20% sample) used to prove model accuracy (Base Case: 18 users saved).
- **Tableau Dashboard:** Represents the **Immediate Action Plan**, identifying **107 high-priority targets** for the upcoming billing cycle with a probability-weighted risk of **$35,392**.

## 4. How to Run the Project

### Phase 1: Data Preparation
1. Place raw CSVs in the root directory.
2. Run `python etl/etl_pipeline.py`.
3. Verify files appear in the `/data/` folder.

### Phase 2: Predictive Modeling
1. Run `python analysis/analysis.py`.
2. Review the terminal for Financial Impact Projections.
3. Check `/dashboard/dashboard_screenshots/` for updated model performance charts.

### Phase 3: Operational Strategy
1. Open `Churn_Analysis_Dashboard.twbx` in Tableau.
2. Use the **"Predicted Churn" Global Filter** to generate the "Call List" for the retention team.

## 5. Project Structure
/
├── etl/
│   └── etl_pipeline.py       # Data cleaning & feature engineering
├── analysis/
│   └── analysis.py          # ML Model, Visuals, & Financial Projections
├── data/
│   ├── dim_users_enriched.csv
│   ├── fact_user_weekly.csv
│   └── model_churn_dataset.csv
├── dashboard/
│   ├── Churn_Dashboard.twbx  # Interactive Production Views
│   └── dashboard_screenshots/# Automated PNG exports from Python
└── final_story/
    └── final_deck.pdf        # Strategic Business Case & Recommendations