# ⚡ OmniPulse: Enterprise E-Commerce Big Data Analytics Platform

A complete, production-ready Big Data Analytics (BDA) project featuring a **23.6 MB** transactional dataset, **Hadoop MapReduce streaming jobs**, **PySpark & Columnar Parquet ETL**, **Spark SQL analytics**, **Machine Learning pipelines (RFM K-Means & Risk Random Forest)**, **MongoDB NoSQL CRUD & aggregation**, **HDFS replication/block-size experiments**, **Apache Hive multi-format analytics**, **Spark Streaming with tumbling & sliding windows**, and an **interactive Streamlit dark-mode web dashboard**.

---

## 🌟 Key Highlights
- 📊 **Big Dataset**: 125,000+ detailed transactional records across 22 attributes (23.63 MB).
- ⚡ **Hadoop MapReduce Engine**: Python streaming MapReduce implementations for Category Sales, Customer RFM, and Payment Fraud Risk detection.
- 🚀 **Spark & Parquet Processing**: Scalable schema validation, distributed feature extraction, and high-performance Snappy Parquet storage.
- 🔍 **Advanced SQL Analytics**: 6+ complex SQL queries computing profitability, cohort retention, and window growth dynamics.
- 🤖 **Machine Learning**: 
  - *Unsupervised*: Customer RFM segmentation via K-Means & Silhouette scoring (Champions, Loyalists, Potential, At-Risk).
  - *Supervised*: Transaction Return Risk classification using Random Forest (93.8% accuracy).
- 📈 **Automated Visual EDA**: 8 publication-ready analytical charts exported in `eda/figures/`.
- 🌐 **Interactive Dashboard**: Modern Streamlit web application with real-time KPI cockpit, MapReduce sandbox, SQL workbench, and live ML inference.
- 🍃 **MongoDB NoSQL Engine**: Document schema design, full CRUD operations, aggregation pipelines, sharding & replication configuration, and performance benchmarks.
- 📂 **HDFS Experiments**: Replication factor (1/2/3) and block size (64/128/256 MB) experiments with throughput analysis.
- 🐝 **Apache Hive Analytics**: HiveQL DDL for TextFile/RCFile/ORC/Parquet, storage format benchmarks, and Hive vs MongoDB comparison.
- 🌊 **Spark Streaming Pipeline**: Real-time event processing with tumbling & sliding window operations, accuracy/latency comparison.

---

## 🎯 Course PI Coverage

| PI | Requirement | Module |
|---|---|---|
| **1.4.2.1** | MongoDB NoSQL + CRUD + Sharding/Replication | `mongodb/` |
| **2.2.1.1** | HDFS Replication/Block-Size Experiments | `hdfs/` |
| **2.2.4.1** | Hive RCFile/ORC vs NoSQL Comparison | `hive/` |
| **3.2.2.1** | Streaming Pipeline (Source → Sink) | `streaming/` |
| **3.3.2.1** | Tumbling vs Sliding Window Comparison | `streaming/` |
| **5.2.1.1** | Visualization of Hive/Spark Results | `eda/` + `dashboard/` |
| **9.1.2.1** | End-to-End Documentation | `docs/` |

---

## 🚀 Quickstart

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Entire Big Data Pipeline (10 stages)**:
   ```bash
   python run_all.py
   ```

3. **Launch the Web Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

4. **Run Individual Modules**:
   ```bash
   # MongoDB NoSQL Operations
   python mongodb/mongodb_operations.py

   # HDFS Experiments
   python hdfs/hdfs_experiments.py

   # Hive Analytics
   python hive/hive_analytics.py

   # Streaming Pipeline
   python streaming/spark_streaming_pipeline.py
   ```

---

## 📁 Project Structure

```
bda_project/
├── data_generator.py              # Synthetic 125K-record dataset generator
├── run_all.py                     # 10-stage master pipeline orchestrator
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/
│   ├── raw/                       # Raw CSV dataset (~23.6 MB)
│   ├── processed/                 # Parquet, CSVs, benchmark results
│   └── streaming/                 # Streaming micro-batch files
│
├── spark/
│   ├── spark_etl.py               # PySpark ETL & Parquet pipeline
│   └── spark_sql_analytics.py     # 6 complex SQL analytical queries
│
├── mapreduce/
│   ├── run_mapreduce_local.py     # MapReduce pipeline simulator
│   ├── 1_sales_by_category/       # Category revenue mapper/reducer
│   ├── 2_customer_rfm/            # Customer RFM mapper/reducer
│   └── 3_payment_fraud_anomaly/   # Fraud detection mapper/reducer
│
├── ml/
│   ├── customer_segmentation.py   # K-Means RFM clustering
│   ├── predictive_models.py       # Random Forest risk classifier
│   └── model_artifacts/           # Saved model pipelines
│
├── eda/
│   ├── run_eda.py                 # 8 automated EDA visualizations
│   └── figures/                   # High-res PNG charts
│
├── mongodb/
│   └── mongodb_operations.py      # NoSQL CRUD, aggregation, sharding
│
├── hdfs/
│   └── hdfs_experiments.py        # Replication & block-size experiments
│
├── hive/
│   └── hive_analytics.py          # HiveQL DDL, benchmarks, comparisons
│
├── streaming/
│   ├── stream_producer.py         # Real-time data stream producer
│   └── spark_streaming_pipeline.py # Windowed streaming analytics
│
├── dashboard/
│   └── app.py                     # Streamlit web dashboard (8 tabs)
│
├── screenshots/                   # Dashboard UI screenshots & execution logs
│
└── docs/
    ├── PROJECT_REPORT.md          # Academic project report
    ├── ARCHITECTURE.md            # System architecture & dataflow
    └── EXECUTION_GUIDE.md         # Setup & execution instructions
```

---

## 📚 Documentation
- [Project Report (Academic / IEEE Format)](docs/PROJECT_REPORT.md)
- [System Architecture & Dataflow](docs/ARCHITECTURE.md)
- [Execution & Hadoop Guide](docs/EXECUTION_GUIDE.md)
