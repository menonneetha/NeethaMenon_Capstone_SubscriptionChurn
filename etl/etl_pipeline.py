import pandas as pd
import numpy as np
from datetime import timedelta
import os

def run_retention_pipeline():
    print("--- Starting ETL Pipeline ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 1. LOAD RAW DATA
    try:
        users = pd.read_csv('users.csv')
        subs = pd.read_csv('subscriptions.csv')
        usage = pd.read_csv('usage_daily.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure raw CSVs are in the /etl/ folder.")
        return

    # 2. STANDARDIZATION
    for df in [users, subs, usage]:
        df.columns = df.columns.str.lower().str.strip()
    
    usage['date'] = pd.to_datetime(usage['date'])
    users['signup_date'] = pd.to_datetime(users['signup_date'])

    # 3. GENERATE: dim_users_enriched.csv
    total_usage = usage.groupby('user_id')['minutes_used'].sum().reset_index()
    total_usage['engagement_band'] = pd.qcut(total_usage['minutes_used'], 3, labels=['Low', 'Med', 'High'], duplicates='drop')
    dim_users = users.merge(total_usage[['user_id', 'engagement_band']], on='user_id', how='left')
    dim_users.to_csv(os.path.join(data_dir, 'dim_users_enriched.csv'), index=False)
    print("Generated: dim_users_enriched.csv")

    # 4. GENERATE: model_churn_dataset.csv
    as_of_date = usage['date'].max()
    l7d_start = as_of_date - timedelta(days=7)
    recent = usage[usage['date'] >= (as_of_date - timedelta(days=28))]
    u_l7d = recent[recent['date'] >= l7d_start].groupby('user_id')['minutes_used'].sum()
    u_prev = recent[recent['date'] < l7d_start].groupby('user_id')['minutes_used'].sum()
    model_df = pd.DataFrame({'l7d_total': u_l7d, 'prev_avg': u_prev / 3}).fillna(0)
    model_df['usage_trend'] = (model_df['l7d_total'] + 0.1) / (model_df['prev_avg'] + 0.1)
    is_churned = subs[subs['status'].str.lower() == 'cancelled']['user_id'].unique()
    model_df['will_churn'] = model_df.index.isin(is_churned).astype(int)
    model_df = model_df.reset_index().rename(columns={'index': 'user_id'})
    model_df.to_csv(os.path.join(data_dir, 'model_churn_dataset.csv'), index=False)
    print("Generated: model_churn_dataset.csv")

    # 5. NEW GENERATION: fact_user_weekly.csv (Required for Cohorts)
    # This groups daily usage into weekly buckets
    usage['week_start'] = usage['date'].dt.to_period('W').apply(lambda r: r.start_time)
    fact_weekly = usage.groupby(['user_id', 'week_start'])['minutes_used'].agg(['sum', 'count']).reset_index()
    fact_weekly.columns = ['user_id', 'week_start', 'weekly_minutes', 'days_active']
    
    # Add cohort (signup month) to each weekly row for easy Tableau filtering
    fact_weekly = fact_weekly.merge(users[['user_id', 'signup_date']], on='user_id', how='left')
    fact_weekly['cohort_month'] = fact_weekly['signup_date'].dt.to_period('M').apply(lambda r: r.start_time)
    
    fact_weekly.to_csv(os.path.join(data_dir, 'fact_user_weekly.csv'), index=False)
    print("Generated: fact_user_weekly.csv")
    
    print(f"--- SUCCESS: All 3 files generated in /data/ ---")

if __name__ == "__main__":
    run_retention_pipeline()
