"""
model.py
DiseaseSpread — SRM IST Mini Project 2026
-----------------------------------------
Random Forest model to predict disease case
surges using real EpiClim + Census features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, roc_auc_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def create_surge_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary surge label forecasting exactly 1 month into the future."""
    df_model = df.copy()
    
    # 1. Convert current rows to actual calendar dates
    df_model['date'] = pd.to_datetime(df_model[['year', 'mon', 'day']].rename(columns={'mon': 'month'}), errors='coerce')
    df_model = df_model.dropna(subset=['date'])
    
    # 2. Calculate 1 month into the future
    df_model['future_date'] = df_model['date'] + pd.DateOffset(months=1)
    df_model['future_year'] = df_model['future_date'].dt.year
    df_model['future_mon']  = df_model['future_date'].dt.month
    
    # 3. Find the cases that actually happened in that specific future month
    monthly_cases = df_model.groupby(['district', 'disease', 'year', 'mon'])['cases'].sum().reset_index()
    monthly_cases.columns = ['district', 'disease', 'future_year', 'future_mon', 'actual_future_cases']
    
    # 4. Merge them! If the future month had no outbreak, fill with 0 cases.
    df_model = df_model.merge(monthly_cases, on=['district','disease','future_year', 'future_mon'], how='left')
    df_model['actual_future_cases'] = df_model['actual_future_cases'].fillna(0)
    
    # Calculate threshold based on historical data
    stats = df_model.groupby(['district','disease'])['cases'].agg(['mean','std']).reset_index()
    stats.columns = ['district','disease','case_mean','case_std']
    # fillna for std if only 1 occurrence
    stats['case_std'] = stats['case_std'].fillna(0)
    
    df_model = df_model.merge(stats, on=['district','disease'], how='left')
    
    # Compare the actual future forecast cases to the threshold
    df_model['surge_threshold'] = df_model['case_mean'] + 0.5 * df_model['case_std']
    df_model['surge'] = (df_model['actual_future_cases'] > df_model['surge_threshold']).astype(int)
    
    df_model = df_model.dropna(subset=['actual_future_cases', 'surge'])
    
    dist = df_model['surge'].value_counts().to_dict()
    print(f"[model] Surge distribution → Normal: {dist.get(0,0)}, Surge: {dist.get(1,0)}")
    return df_model


def prepare_features(df: pd.DataFrame):
    """Encode categorical features and prepare train split."""
    # Note: handle both 'Temp' and 'temp_celsius' based on which stage of preprocessing data is in!
    feature_cols = ['Temp', 'temp_celsius', 'rainfall_mm', 'leaf_area_index', 'mon', 'cases']
    available = [c for c in feature_cols if c in df.columns]
    
    # Dummy encoding for district and disease
    df_clean = pd.get_dummies(df, columns=['district', 'disease'])
    
    # Grab generated column names
    encoded_cols = [c for c in df_clean.columns if c.startswith('district_') or c.startswith('disease_')]
    final_features = available + encoded_cols
    
    df_clean = df_clean[final_features + ['surge']].dropna()
    X = df_clean[final_features]
    y = df_clean['surge']
    
    print(f"[model] Training rows : {len(X)}")
    return X, y, final_features


def train_model(X, y, test_size=0.2, random_state=42):
    """Train Random Forest with SMOTE and Threshold Tuning."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=12,
        random_state=random_state
    )
    model.fit(X_train_balanced, y_train_balanced)
    
    # Get raw probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Predict using a 20% custom threshold
    threshold = 0.20
    y_pred = (y_prob >= threshold).astype(int)

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report  = classification_report(y_test, y_pred, target_names=['Normal','Surge'])
    conf    = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"🌲 Random Forest Results (Threshold: {threshold})")
    print(f"{'='*50}")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf}")
    
    return {
        'model': model, 'accuracy': acc, 'roc_auc': roc_auc,
        'report': report, 'conf_matrix': conf,
        'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred
    }


def save_artifacts(model, features, output_dir='outputs'):
    """Save the model and generate feature importance chart."""
    import joblib
    
    model_dir = os.path.join(output_dir, 'results')
    graph_dir = os.path.join(output_dir, 'graphs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    
    # 1. Save Model
    model_path = os.path.join(model_dir, 'rf_model.pkl')
    joblib.dump(model, model_path)
    print(f"✅ Model saved → {model_path}")
    
    # 2. Generate Graph
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(importance['feature'], importance['importance'], color='#3498db', edgecolor='black')
    ax.set_title('🔑 Top 10 Disease Predictors — Random Forest', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    
    graph_path = os.path.join(graph_dir, 'feature_importance.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Graph saved → {graph_path}")


if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'merged_clean.csv'

    print("DiseaseSpread — Model Training")
    print("=" * 50)
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: {data_path} not found. Run preprocessing.py first!")
        sys.exit(1)
        
    df = create_surge_label(df)
    X, y, feats = prepare_features(df)
    results = train_model(X, y)
    save_artifacts(results['model'], feats)