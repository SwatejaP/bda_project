"""
OmniPulse Big Data Analytics: Spark SQL & Advanced Aggregation Engine
Executes complex distributed SQL queries including window functions, cohort retention,
and financial KPIs across the processed dataset.
"""

import os
import sys
import time
import pandas as pd

def run_sql_analytics():
    print("\n=======================================================")
    print("[*] Executing Advanced Big Data SQL Analytics Queries")
    print("=======================================================")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parquet_path = os.path.join(base_dir, "data", "processed", "cleaned_transactions.parquet")
    csv_path = os.path.join(base_dir, "data", "processed", "cleaned_transactions.csv")
    
    if not os.path.exists(parquet_path) and not os.path.exists(csv_path):
        from spark.spark_etl import run_spark_etl
        run_spark_etl()

    # Load data for SQL execution
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(csv_path)

    # Use DuckDB for ultra-fast in-memory vectorized SQL execution or PySpark SQL
    import sqlite3
    
    conn = sqlite3.connect(":memory:")
    
    # SQLite is case-insensitive for column names. 
    # Drop any lower-case duplicates (like product_id vs Product_ID)
    cols_to_keep = []
    seen = set()
    for c in df.columns:
        if c.lower() not in seen:
            seen.add(c.lower())
            cols_to_keep.append(c)
    df = df[cols_to_keep]

    df.to_sql("transactions", conn, index=False, if_exists="replace")
    
    queries = [
        {
            "title": "Query 1: Category Revenue & Profitability Matrix",
            "desc": "Aggregates total revenue, units sold, average discount, and estimated net profit margin by category.",
            "sql": """
                SELECT 
                    Product_Category,
                    COUNT(*) AS Total_Orders,
                    SUM(Quantity) AS Total_Units_Sold,
                    ROUND(SUM(Total_Amount), 2) AS Total_Revenue_USD,
                    ROUND(AVG(Total_Amount), 2) AS Avg_Order_Value_USD,
                    ROUND(AVG(Discount_Percent) * 100, 2) AS Avg_Discount_Pct,
                    ROUND(SUM(Net_Profit_Estimate), 2) AS Estimated_Net_Profit_USD,
                    ROUND((SUM(Net_Profit_Estimate) / SUM(Total_Amount)) * 100, 2) AS Profit_Margin_Pct
                FROM transactions
                GROUP BY Product_Category
                ORDER BY Total_Revenue_USD DESC;
            """
        },
        {
            "title": "Query 2: Regional Market Penetration & Average Basket Value",
            "desc": "Analyzes market volume, revenue contribution, and delivery success rates across continents.",
            "sql": """
                SELECT 
                    Customer_Region,
                    COUNT(DISTINCT Customer_ID) AS Unique_Customers,
                    COUNT(*) AS Total_Transactions,
                    ROUND(SUM(Total_Amount), 2) AS Regional_Revenue_USD,
                    ROUND(AVG(Total_Amount), 2) AS Avg_Basket_Size_USD,
                    ROUND(SUM(CASE WHEN Delivery_Status = 'Delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Delivery_Success_Pct,
                    ROUND(SUM(CASE WHEN Delivery_Status = 'Returned' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Return_Rate_Pct
                FROM transactions
                GROUP BY Customer_Region
                ORDER BY Regional_Revenue_USD DESC;
            """
        },
        {
            "title": "Query 3: Payment Gateway Security & Fraud Anomaly Risk",
            "desc": "Calculates risk exposure and fraud loss across all payment gateways.",
            "sql": """
                SELECT 
                    Payment_Method,
                    COUNT(*) AS Total_Txns,
                    SUM(Is_Fraudulent) AS Fraud_Txns,
                    ROUND(AVG(Is_Fraudulent) * 100, 2) AS Fraud_Rate_Pct,
                    ROUND(SUM(Total_Amount), 2) AS Total_Volume_USD,
                    ROUND(SUM(CASE WHEN Is_Fraudulent = 1 THEN Total_Amount ELSE 0 END), 2) AS Total_Fraud_Loss_USD,
                    ROUND(AVG(CASE WHEN Is_Fraudulent = 1 THEN Total_Amount ELSE NULL END), 2) AS Avg_Fraudulent_Txn_USD
                FROM transactions
                GROUP BY Payment_Method
                ORDER BY Total_Fraud_Loss_USD DESC;
            """
        },
        {
            "title": "Query 4: Top 10 High-Value VIP Customers (Lifetime Spend & Activity)",
            "desc": "Identifies the highest grossing customers with their tenure, order count, and ratings.",
            "sql": """
                SELECT 
                    Customer_ID,
                    Customer_Region,
                    Country,
                    Customer_Age,
                    Customer_Gender,
                    COUNT(*) AS Total_Orders,
                    ROUND(SUM(Total_Amount), 2) AS Lifetime_Value_USD,
                    ROUND(AVG(Total_Amount), 2) AS Avg_Spend_Per_Order,
                    ROUND(AVG(Customer_Rating), 2) AS Avg_Satisfaction_Rating
                FROM transactions
                GROUP BY Customer_ID, Customer_Region, Country, Customer_Age, Customer_Gender
                ORDER BY Lifetime_Value_USD DESC
                LIMIT 10;
            """
        },
        {
            "title": "Query 5: Time-Series Monthly Revenue Velocity & Order Dynamics",
            "desc": "Analyzes month-over-month revenue trends across 2024 and 2025.",
            "sql": """
                SELECT 
                    Order_Year,
                    Order_Month,
                    COUNT(*) AS Monthly_Orders,
                    SUM(Quantity) AS Monthly_Units_Sold,
                    ROUND(SUM(Total_Amount), 2) AS Monthly_Revenue_USD,
                    ROUND(AVG(Total_Amount), 2) AS Avg_Order_USD
                FROM transactions
                GROUP BY Order_Year, Order_Month
                ORDER BY Order_Year, Order_Month;
            """
        },
        {
            "title": "Query 6: Device Channel vs. Delivery Fulfillment & Order Priority Matrix",
            "desc": "Measures mobile vs desktop conversion and delivery fulfillment efficiencies.",
            "sql": """
                SELECT 
                    Device_Channel,
                    COUNT(*) AS Total_Orders,
                    ROUND(SUM(Total_Amount), 2) AS Channel_Revenue_USD,
                    ROUND(AVG(Session_Duration_Min), 1) AS Avg_Session_Min,
                    ROUND(SUM(CASE WHEN Order_Priority = 'Express Same-Day' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Express_Priority_Pct,
                    ROUND(SUM(CASE WHEN Delivery_Status = 'Returned' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Return_Pct
                FROM transactions
                GROUP BY Device_Channel
                ORDER BY Channel_Revenue_USD DESC;
            """
        }
    ]

    results_dict = {}

    for q in queries:
        print(f"\n[*] {q['title']}")
        print(f"    Description: {q['desc']}")
        t_start = time.time()
        res_df = pd.read_sql_query(q["sql"], conn)
        t_elapsed = time.time() - t_start
        print(f"    Executed in: {t_elapsed:.4f}s")
        print(res_df.to_string(index=False))
        results_dict[q["title"]] = res_df

    conn.close()
    print("\n[+] All Big Data SQL Analytics queries executed successfully!")
    print("=======================================================\n")
    return results_dict

if __name__ == "__main__":
    run_sql_analytics()
