import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def run_full_analysis():
    print("--- Starting Analysis ---")
    
    # 1. SETUP PATHS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_folder = os.path.join(project_root, "data")
    
    # AUTO-CREATE DIRECTORY
    screenshot_dir = os.path.join(project_root, 'dashboard', 'dashboard_screenshots')
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
        print(f"Created missing directory: {screenshot_dir}")
    
    # 2. LOAD DATA
    input_path = os.path.join(data_folder, 'model_churn_dataset.csv')
    dim_users_path = os.path.join(data_folder, 'dim_users_enriched.csv')
    
    try:
        model_df = pd.read_csv(input_path)
        dim_users = pd.read_csv(dim_users_path)
    except FileNotFoundError as e:
        print(f"Error: Missing files. Please ensure {input_path} and {dim_users_path} exist.")
        return

    # 3. GENERATE INSIGHT VISUALS
    print("Generating Insight Charts...")
    
    # Chart 1: Monthly Trend
    months = ['Jun 25', 'Jul 25', 'Aug 25', 'Sep 25', 'Oct 25', 'Nov 25', 'Dec 25', 'Jan 26']
    rates = [1.1, 1.31, 1.68, 2.34, 3.47, 3.59, 5.79, 7.94]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=months, y=rates, hue=months, palette='Blues_d', legend=False)
    plt.title('Monthly Subscription Churn Rate Trend (%)', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(screenshot_dir, 'monthly_churn_trend.png'))
    plt.close()
    print("Saved: monthly_churn_trend.png")

    # Chart 2: Churn by Engagement (Slide 8)
    # Merging churn data back to enriched users to get the engagement band
    temp_merge = dim_users.merge(model_df[['user_id', 'will_churn']], on='user_id', how='left').fillna(0)
    plt.figure(figsize=(8, 5))
    engagement_stats = temp_merge.groupby('engagement_band')['will_churn'].mean() * 100
    engagement_stats.plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99'], edgecolor='black')
    plt.title('Churn Rate (%) by Engagement Band', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(screenshot_dir, 'churn_by_engagement.png'))
    plt.close()
    print("Saved: churn_by_engagement.png")

    # Chart 3: Usage Trend Analysis (Slide 10)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=model_df[model_df['will_churn'] == 1], x='usage_trend', label='Churned', fill=True, color='red')
    sns.kdeplot(data=model_df[model_df['will_churn'] == 0], x='usage_trend', label='Retained', fill=True, color='green')
    plt.title('Usage Trend Signal: Churned vs. Retained Users')
    plt.xlabel('Usage Trend (1.0 = Stable, < 1.0 = Declining)')
    plt.legend()
    plt.savefig(os.path.join(screenshot_dir, 'usage_trend_analysis.png'))
    plt.close()
    print("Saved: usage_trend_analysis.png")

    # 4. MODEL TRAINING (Maintaining today's Tableau Logic)
    X = model_df.drop(columns=['user_id', 'will_churn'])
    y = model_df['will_churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 5. GENERATE EVALUATION CHARTS
    
    # Chart 4: Feature Importance
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='skyblue')
    plt.title('Feature Importance: What Drives Churn?')
    plt.savefig(os.path.join(screenshot_dir, 'feature_importance.png'))
    plt.close()
    print("Saved: feature_importance.png")

    # Chart 5: Confusion Matrix
    probs = rf.predict_proba(X_test)[:, 1]
    y_pred = (probs >= 0.3).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('Confusion Matrix (Threshold 0.3)')
    plt.savefig(os.path.join(screenshot_dir, 'confusion_matrix.png'))
    plt.close()
    print("Saved: confusion_matrix.png")

    # 6. CONSTRUCT FINAL TABLEAU OUTPUT
    output = X_test.copy()
    output['user_id'] = model_df.loc[X_test.index, 'user_id']
    output['will_churn'] = y_test.values
    output['probability_score'] = probs
    output['predicted_churn'] = y_pred
    output['revenue_at_risk_estimate'] = probs * 455.0
    output.to_csv(os.path.join(data_folder, 'tableau_final_output.csv'), index=False)
    
    # 7. FINANCIAL IMPACT
    total_churners = model_df[model_df['will_churn'] == 1].shape[0]
    print(f"\n--- Final Projections (Targeting {total_churners} users) ---")
    for name, rate in [("Worst Case", 0.10), ("Base Case", 0.20), ("Best Case", 0.35)]:
        saved = int(total_churners * rate)
        net_recovered = (saved * 455) - (saved * (455 * 0.15))
        print(f"{name}: {saved} users | Net Recovered: ${net_recovered:,.2f}")

if __name__ == "__main__":
    run_full_analysis()
