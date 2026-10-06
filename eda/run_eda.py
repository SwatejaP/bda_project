"""
OmniPulse Big Data Analytics: Automated EDA & Statistical Visualization Engine
Generates publication-quality charts and figures for reports and presentations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Configure dark/modern styling
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def run_eda_visualizations():
    print("\n=======================================================")
    print("[*] Generating High-Resolution Visual Analytics & Figures")
    print("=======================================================")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parquet_path = os.path.join(base_dir, "data", "processed", "cleaned_transactions.parquet")
    csv_path = os.path.join(base_dir, "data", "raw", "ecommerce_transactions.csv")
    rfm_path = os.path.join(base_dir, "data", "processed", "rfm_segmented_customers.csv")
    fig_dir = os.path.join(base_dir, "eda", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(csv_path)
        df["Order_Timestamp"] = pd.to_datetime(df["Order_Date"])
        df["Order_Year"] = df["Order_Timestamp"].dt.year
        df["Order_Month"] = df["Order_Timestamp"].dt.month

    # Figure 1: Revenue by Product Category
    print("[*] Creating Fig 1: Revenue & Units by Product Category...")
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    cat_summary = df.groupby("Product_Category").agg(
        Total_Revenue=("Total_Amount", "sum"),
        Total_Units=("Quantity", "sum")
    ).sort_values("Total_Revenue", ascending=False).reset_index()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bars = ax1.bar(cat_summary["Product_Category"], cat_summary["Total_Revenue"] / 1e6, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Total Revenue ($ Millions)", fontsize=12, fontweight="bold", color="#1a1a1a")
    ax1.set_title("Total Revenue & Units Sold by Product Category", fontsize=14, fontweight="bold", pad=15)
    ax1.tick_params(axis="x", rotation=15, labelsize=10)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.1fM"))

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"${height:.2f}M",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "1_revenue_by_category.png")
    plt.savefig(fig1_path)
    plt.close()

    # Figure 2: Time-Series Monthly Revenue Velocity (2024-2025)
    print("[*] Creating Fig 2: Monthly Revenue Velocity Trend...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    df["YearMonth"] = pd.to_datetime(df["Order_Date"]).dt.to_period("M").astype(str)
    monthly_rev = df.groupby("YearMonth")["Total_Amount"].sum() / 1e6

    ax.plot(monthly_rev.index, monthly_rev.values, marker="o", color="#007acc", linewidth=2.5, markersize=6, label="Monthly Gross Revenue ($M)")
    ax.fill_between(monthly_rev.index, monthly_rev.values, color="#007acc", alpha=0.15)
    ax.set_title("Monthly Revenue Trajectory (2024 - 2025)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Gross Revenue ($ Millions)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year-Month", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2fM"))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "2_monthly_sales_trend.png")
    plt.savefig(fig2_path)
    plt.close()

    # Figure 3: Regional Revenue Distribution (Donut & Basket Size)
    print("[*] Creating Fig 3: Regional Sales Distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    reg_summary = df.groupby("Customer_Region").agg(
        Revenue=("Total_Amount", "sum"),
        Avg_Order=("Total_Amount", "mean")
    ).sort_values("Revenue", ascending=False)

    palette = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f"]
    wedges, texts, autotexts = ax1.pie(
        reg_summary["Revenue"], labels=reg_summary.index, autopct="%1.1f%%",
        colors=palette, startangle=140, pctdistance=0.75,
        textprops=dict(color="#111", fontweight="bold", fontsize=9)
    )
    # Donut hole
    centre_circle = plt.Circle((0,0), 0.55, fc="white")
    ax1.add_artist(centre_circle)
    ax1.set_title("Revenue Share by Geographic Region", fontsize=13, fontweight="bold")

    # Basket size bar chart
    ax2.barh(reg_summary.index, reg_summary["Avg_Order"], color=palette, edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Average Order Value (USD)", fontsize=11, fontweight="bold")
    ax2.set_title("Average Basket Size by Region", fontsize=13, fontweight="bold")
    for i, v in enumerate(reg_summary["Avg_Order"]):
        ax2.text(v + 5, i, f"${v:.2f}", va="center", fontweight="bold", fontsize=9)
    
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, "3_regional_distribution.png")
    plt.savefig(fig3_path)
    plt.close()

    # Figure 4: Payment Fraud & Risk Analysis
    print("[*] Creating Fig 4: Payment Fraud & Loss Exposure...")
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    pay_summary = df.groupby("Payment_Method").agg(
        Total_Volume=("Total_Amount", "sum"),
        Fraud_Loss=("Total_Amount", lambda x: x[df.loc[x.index, "Is_Fraudulent"] == 1].sum()),
        Fraud_Rate=("Is_Fraudulent", lambda x: x.mean() * 100)
    ).sort_values("Total_Volume", ascending=False).reset_index()

    x = np.arange(len(pay_summary))
    width = 0.35

    ax1.bar(x - width/2, pay_summary["Total_Volume"] / 1e6, width, label="Total Volume ($M)", color="#3366cc", alpha=0.85)
    ax1.set_ylabel("Total Transaction Volume ($M)", fontsize=11, fontweight="bold", color="#3366cc")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pay_summary["Payment_Method"], rotation=15, fontsize=10)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.1fM"))

    ax2 = ax1.twinx()
    ax2.plot(x, pay_summary["Fraud_Rate"], color="#dc3912", marker="s", linewidth=2.5, markersize=8, label="Fraud Rate (%)")
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight="bold", color="#dc3912")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f%%"))

    plt.title("Payment Gateway Volume vs. Fraud Anomaly Exposure", fontsize=14, fontweight="bold", pad=15)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    plt.tight_layout()
    fig4_path = os.path.join(fig_dir, "4_payment_fraud_analysis.png")
    plt.savefig(fig4_path)
    plt.close()

    # Figure 5: Customer RFM Clusters
    if os.path.exists(rfm_path):
        print("[*] Creating Fig 5: RFM Customer Clusters...")
        rfm_df = pd.read_csv(rfm_path)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        segments = rfm_df["Customer_Segment"].unique()
        colors = {"Champions (VIP)": "#2ca02c", "Loyal Customers": "#1f77b4", "Potential / Growth": "#ff7f0e", "At-Risk / Hibernating": "#d62728"}
        
        for seg in segments:
            sub = rfm_df[rfm_df["Customer_Segment"] == seg]
            ax.scatter(sub["Recency_Days"], sub["Monetary_Total_USD"], 
                       label=seg, alpha=0.65, edgecolors="none", s=28, color=colors.get(seg, "#333333"))

        ax.set_yscale("log")
        ax.set_xlabel("Recency (Days Since Last Purchase)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Total Lifetime Spend (USD - Log Scale)", fontsize=11, fontweight="bold")
        ax.set_title("Customer RFM Segmentation Cohorts (K-Means Clustering)", fontsize=14, fontweight="bold", pad=15)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%d"))
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        fig5_path = os.path.join(fig_dir, "5_rfm_customer_clusters.png")
        plt.savefig(fig5_path)
        plt.close()

    # Figure 6: Order Priority vs. Delivery Status
    print("[*] Creating Fig 6: Delivery Status by Order Priority...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    cross_tab = pd.crosstab(df["Order_Priority"], df["Delivery_Status"], normalize="index") * 100
    cross_tab.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c", edgecolor="black", alpha=0.85)
    ax.set_title("Fulfillment & Delivery Status Breakdown by Order Priority", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Order Priority Tier", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Delivery Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig6_path = os.path.join(fig_dir, "6_order_priority_delivery_status.png")
    plt.savefig(fig6_path)
    plt.close()

    # Figure 7: Correlation Heatmap
    print("[*] Creating Fig 7: Correlation Matrix Heatmap...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    num_cols = ["Customer_Age", "Unit_Price", "Quantity", "Discount_Percent", "Shipping_Cost", "Total_Amount", "Customer_Rating", "Session_Duration_Min", "Is_Fraudulent"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig7_path = os.path.join(fig_dir, "7_correlation_heatmap.png")
    plt.savefig(fig7_path)
    plt.close()

    # Figure 8: Device Channel Share
    print("[*] Creating Fig 8: Device Channel Distribution...")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    chan_summary = df["Device_Channel"].value_counts()
    bars = ax.barh(chan_summary.index, chan_summary.values, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Transaction Count", fontsize=11, fontweight="bold")
    ax.set_title("Device & Channel Ingestion Distribution", fontsize=13, fontweight="bold")
    for bar in bars:
        width = bar.get_width()
        pct = (width / len(df)) * 100
        ax.text(width + 500, bar.get_y() + bar.get_height()/2, f"{width:,} ({pct:.1f}%)", va="center", fontweight="bold", fontsize=9)
    ax.set_xlim(0, max(chan_summary.values) * 1.18)
    plt.tight_layout()
    fig8_path = os.path.join(fig_dir, "8_device_channel_share.png")
    plt.savefig(fig8_path)
    plt.close()

    print(f"[+] All 8 Analytical Visualizations saved successfully to: {fig_dir}")
    print("=======================================================\n")

if __name__ == "__main__":
    run_eda_visualizations()
