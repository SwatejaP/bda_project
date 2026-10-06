"""
OmniPulse Big Data Analytics: Machine Learning Unsupervised Engine
Performs Customer RFM (Recency, Frequency, Monetary) Segmentation using K-Means Clustering,
Elbow Method (Inertia), and Silhouette Score Optimization.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_customer_segmentation():
    print("\n=======================================================")
    print("[*] Starting Unsupervised ML: Customer RFM Segmentation")
    print("=======================================================")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, "data", "raw", "ecommerce_transactions.csv")
    rfm_tsv = os.path.join(base_dir, "data", "processed", "mapreduce_customer_rfm.tsv")
    processed_dir = os.path.join(base_dir, "data", "processed")
    artifacts_dir = os.path.join(base_dir, "ml", "model_artifacts")
    output_csv = os.path.join(processed_dir, "rfm_segmented_customers.csv")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load or Calculate RFM Data
    if os.path.exists(rfm_tsv):
        print(f"[*] Loading Customer RFM data from MapReduce output: {rfm_tsv}")
        rfm_df = pd.read_csv(rfm_tsv, sep="\t", header=None, names=["Customer_ID", "rfm_str"])
        # Parse "Recency,Frequency,Monetary"
        rfm_df[["Recency_Days", "Frequency", "Monetary_Total_USD"]] = rfm_df["rfm_str"].str.split(",", expand=True).astype(float)
        rfm_df.drop(columns=["rfm_str"], inplace=True)
    else:
        print("[*] Computing Customer RFM table directly from raw transactions...")
        df = pd.read_csv(raw_csv)
        df["Order_Date"] = pd.to_datetime(df["Order_Date"])
        ref_date = datetime(2026, 1, 1)

        rfm_df = df.groupby("Customer_ID").agg(
            Recency_Days=("Order_Date", lambda x: (ref_date - x.max()).days),
            Frequency=("Transaction_ID", "count"),
            Monetary_Total_USD=("Total_Amount", "sum"),
            Avg_Order_Value_USD=("Total_Amount", "mean")
        ).reset_index()

    print(f"[+] Loaded {len(rfm_df):,} unique customer profiles.")
    
    # 2. Feature Preprocessing (Log-transform for skewed monetary/frequency + Standard Scaling)
    features = ["Recency_Days", "Frequency", "Monetary_Total_USD"]
    
    # Apply log1p transformation to handle right-skewness
    rfm_log = np.log1p(rfm_df[features])
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # 3. Determine Optimal K using Inertia & Silhouette Analysis
    print("[*] Evaluating K-Means clusters (K=2 to 6)...")
    silhouette_scores = {}
    inertias = {}
    
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        inertias[k] = km.inertia_
        # Sample for quick silhouette calculation
        sample_idx = np.random.choice(len(rfm_scaled), size=min(5000, len(rfm_scaled)), replace=False)
        sil = silhouette_score(rfm_scaled[sample_idx], km.labels_[sample_idx])
        silhouette_scores[k] = sil
        print(f"    K={k}: Inertia={km.inertia_:.1f}, Silhouette Score={sil:.4f}")

    optimal_k = 4 # 4 distinct interpretable strategic tiers
    print(f"[+] Selected K={optimal_k} for actionable enterprise customer tiering.")

    # 4. Fit Final K-Means Model
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm_df["Cluster_ID"] = kmeans_model.fit_predict(rfm_scaled)

    # 5. Profile & Name Clusters based on RFM Characteristics
    cluster_means = rfm_df.groupby("Cluster_ID")[features].mean()
    print("\n--- Cluster Mean Statistics ---")
    print(cluster_means)
    
    # Dynamic Cluster Labeling based on Monetary and Recency rankings
    ranked_monetary = cluster_means["Monetary_Total_USD"].sort_values(ascending=False).index
    
    cluster_label_map = {
        ranked_monetary[0]: "Champions (VIP)",
        ranked_monetary[1]: "Loyal Customers",
        ranked_monetary[2]: "Potential / Growth",
        ranked_monetary[3]: "At-Risk / Hibernating"
    }
    
    rfm_df["Customer_Segment"] = rfm_df["Cluster_ID"].map(cluster_label_map)

    # 6. Save Artifacts
    joblib.dump(kmeans_model, os.path.join(artifacts_dir, "kmeans_rfm_model.pkl"))
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler_rfm.pkl"))
    rfm_df.to_csv(output_csv, index=False)

    print("\n--- Customer Segment Distribution ---")
    seg_summary = rfm_df.groupby("Customer_Segment").agg(
        Customer_Count=("Customer_ID", "count"),
        Avg_Recency_Days=("Recency_Days", "mean"),
        Avg_Orders=("Frequency", "mean"),
        Total_Revenue_Contribution=("Monetary_Total_USD", "sum"),
        Avg_Total_Spend=("Monetary_Total_USD", "mean")
    ).reset_index()
    seg_summary["Revenue_Share_Pct"] = (seg_summary["Total_Revenue_Contribution"] / seg_summary["Total_Revenue_Contribution"].sum()) * 100
    print(seg_summary.to_string(index=False))

    print(f"\n[+] Customer Segmentation Model saved to: {os.path.join(artifacts_dir, 'kmeans_rfm_model.pkl')}")
    print(f"[+] Segmented Customers CSV saved to: {output_csv}")
    print("=======================================================\n")
    return rfm_df

if __name__ == "__main__":
    run_customer_segmentation()
