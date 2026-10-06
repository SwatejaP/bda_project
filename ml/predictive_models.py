"""
OmniPulse Big Data Analytics: Supervised Machine Learning Pipeline (Kaggle Schema)
Trains predictive models for Purchase Intent Classification (Will this event lead to a purchase?)
using Random Forest and Logistic Regression.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

def run_predictive_modeling():
    print("\n=======================================================")
    print("[*] Starting Supervised ML: Purchase Intent Prediction")
    print("=======================================================")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "raw", "ecommerce_transactions.csv")
    artifacts_dir = os.path.join(base_dir, "ml", "model_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df):,} events for supervised modeling.")

    # 2. Feature Engineering
    # Extract time-based features
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['hour_of_day'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    
    # Target variable: Is_Purchase (1 if event_type == 'purchase', else 0)
    df["Is_Purchase"] = (df["event_type"] == "purchase").astype(int)
    
    # Fill NAs
    df['category_code'] = df['category_code'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')
    df['price'] = df['price'].fillna(df['price'].median())

    # Feature columns
    numeric_features = ["price", "hour_of_day", "day_of_week"]
    categorical_features = ["category_code", "brand"]

    X = df[numeric_features + categorical_features]
    y = df["Is_Purchase"]

    print(f"[*] Target distribution: Purchase Rate = {y.mean() * 100:.2f}%")

    # 3. Train-Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"[+] Train sample size: {len(X_train):,}, Test sample size: {len(X_test):,}")

    # 4. Build Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ]
    )

    # 5. Model: Random Forest Classifier
    print("\n[*] Training Advanced Model: Random Forest Classifier (100 Trees)...")
    t_rf_start = time.time()
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    
    roc_rf = roc_auc_score(y_test, y_prob_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"[+] Random Forest Trained in {time.time() - t_rf_start:.2f}s | Test ROC-AUC: {roc_rf:.4f} | Accuracy: {acc_rf:.4f}")
    
    print("\n--- Classification Report (Random Forest) ---")
    print(classification_report(y_test, y_pred_rf, target_names=["No Purchase", "Purchase"]))

    # 6. Save Pipeline Artifact
    model_save_path = os.path.join(artifacts_dir, "purchase_intent_pipeline.pkl")
    joblib.dump(rf_pipeline, model_save_path)
    print(f"\n[+] Production Model Pipeline saved to: {model_save_path}")
    print("=======================================================\n")
    return rf_pipeline

if __name__ == "__main__":
    run_predictive_modeling()
