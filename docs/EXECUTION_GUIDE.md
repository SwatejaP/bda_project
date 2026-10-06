# 🚀 OmniPulse Project Execution & Command Guide

This guide provides clear, step-by-step instructions to run each component of the **OmniPulse Big Data Analytics Platform**.

---

## 1. Quickstart (One-Command Execution)

To run the entire end-to-end Big Data pipeline (Data generation -> Spark ETL -> MapReduce jobs -> Spark SQL -> ML Training -> EDA generation):

```bash
cd bda_project
python run_all.py
```

---

## 2. Running the Interactive Streamlit Web Dashboard

Launch the web dashboard to explore the interactive KPI cockpit, run MapReduce jobs from the browser, execute Spark SQL queries, and perform live ML inference:

```bash
cd bda_project
streamlit run dashboard/app.py
```
> The dashboard will automatically open at: `http://localhost:8501`

---

## 3. Running Individual Modules & Hadoop MapReduce Jobs

### A. Data Ingestion & Generation
Generate or regenerate the 125,000 transaction dataset (>23.6 MB):
```bash
python data_generator.py 125000
```

### B. Distributed Spark & Columnar Parquet ETL
Run schema enforcement, null imputation, timestamp feature engineering, and Parquet export:
```bash
python spark/spark_etl.py
```

### C. Hadoop MapReduce Jobs (Local Simulation)
Run all 3 MapReduce streaming jobs:
```bash
python mapreduce/run_mapreduce_local.py
```

To run individual MapReduce jobs via Unix/PowerShell pipes:
```bash
# Job 1: Category Sales Aggregator
cat data/raw/ecommerce_transactions.csv | python mapreduce/1_sales_by_category/mapper.py | sort | python mapreduce/1_sales_by_category/reducer.py

# Job 2: Customer RFM Aggregator
cat data/raw/ecommerce_transactions.csv | python mapreduce/2_customer_rfm/mapper.py | sort | python mapreduce/2_customer_rfm/reducer.py

# Job 3: Payment Fraud Detector
cat data/raw/ecommerce_transactions.csv | python mapreduce/3_payment_fraud_anomaly/mapper.py | sort | python mapreduce/3_payment_fraud_anomaly/reducer.py
```

### D. Native Hadoop Streaming Cluster Execution
If running on an Apache Hadoop cluster:
```bash
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
    -input /user/bda/ecommerce_transactions.csv \
    -output /user/bda/output_category_sales \
    -mapper "python3 mapper.py" \
    -reducer "python3 reducer.py" \
    -file mapreduce/1_sales_by_category/mapper.py \
    -file mapreduce/1_sales_by_category/reducer.py
```

### E. Spark SQL & Advanced Analytics
Execute complex analytical SQL queries:
```bash
python spark/spark_sql_analytics.py
```

### F. Machine Learning Pipelines
1. **Unsupervised Customer RFM Clustering (K-Means)**:
   ```bash
   python ml/customer_segmentation.py
   ```
2. **Supervised Return Risk Classification (Random Forest)**:
   ```bash
   python ml/predictive_models.py
   ```

### G. Automated High-Resolution EDA Figures
Generate publication-quality charts into `eda/figures/`:
```bash
python eda/run_eda.py
```

---

## 4. Directory Structure Summary

```
bda_project/
├── data/
│   ├── raw/
│   │   └── ecommerce_transactions.csv    # 23.63 MB raw dataset
│   └── processed/
│       ├── cleaned_transactions.parquet  # Columnar storage
│       ├── mapreduce_*.tsv               # MapReduce outputs
│       └── rfm_segmented_customers.csv   # Segmented customers
├── mapreduce/                            # Hadoop streaming jobs
│   ├── 1_sales_by_category/
│   ├── 2_customer_rfm/
│   ├── 3_payment_fraud_anomaly/
│   └── run_mapreduce_local.py
├── spark/                                # Spark & SQL pipeline
│   ├── spark_etl.py
│   └── spark_sql_analytics.py
├── ml/                                   # Machine Learning
│   ├── customer_segmentation.py
│   ├── predictive_models.py
│   └── model_artifacts/
├── eda/                                  # Automated EDA
│   ├── run_eda.py
│   └── figures/                          # 8 High-res figures
├── dashboard/
│   └── app.py                            # Streamlit web app
├── docs/                                 # Full documentation
│   ├── PROJECT_REPORT.md
│   ├── ARCHITECTURE.md
│   └── EXECUTION_GUIDE.md
├── requirements.txt
└── run_all.py                            # Master pipeline runner
```
