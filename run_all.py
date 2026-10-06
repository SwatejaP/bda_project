"""
OmniPulse Big Data Analytics: Master Orchestrator Pipeline
Executes the entire end-to-end Big Data workflow:
Data Ingestion -> MapReduce Engine -> Spark ETL -> Spark SQL Analytics ->
Machine Learning -> EDA Reports -> MongoDB NoSQL -> HDFS Experiments ->
Hive Analytics -> Streaming Pipeline
"""

import os
import sys
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_master_pipeline():
    total_start = time.time()
    
    print("=========================================================================")
    print("        [+] OMNIPULSE: ENTERPRISE BIG DATA ANALYTICS PIPELINE")
    print("                 VTU 7th Sem / Production BDA Architecture")
    print("=========================================================================\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    # Step 1: Data Generation
    print("--- [STAGE 1/10] High-Volume Data Ingestion & Generation ---")
    from data_generator import generate_ecommerce_dataset
    raw_csv = os.path.join(base_dir, "data", "raw", "ecommerce_transactions.csv")
    if not os.path.exists(raw_csv):
        generate_ecommerce_dataset(125000, raw_csv)
    else:
        file_size_mb = os.path.getsize(raw_csv) / (1024 * 1024)
        print(f"[+] Raw dataset already exists ({file_size_mb:.2f} MB): {raw_csv}")

    # Step 2: Spark & Parquet ETL
    print("\n--- [STAGE 2/10] Distributed Spark & Parquet ETL Pipeline ---")
    from spark.spark_etl import run_spark_etl
    run_spark_etl()

    # Step 3: Hadoop MapReduce Jobs
    print("\n--- [STAGE 3/10] Hadoop MapReduce Streaming Pipeline ---")
    from mapreduce.run_mapreduce_local import run_all_mapreduce_jobs
    run_all_mapreduce_jobs()

    # Step 4: Spark SQL & Advanced Query Engine
    print("\n--- [STAGE 4/10] Spark SQL & Window Function Analytics ---")
    from spark.spark_sql_analytics import run_sql_analytics
    run_sql_analytics()

    # Step 5: Machine Learning (Unsupervised RFM & Supervised Risk)
    print("\n--- [STAGE 5/10] Distributed Machine Learning Modeling ---")
    from ml.customer_segmentation import run_customer_segmentation
    from ml.predictive_models import run_predictive_modeling
    run_customer_segmentation()
    run_predictive_modeling()

    # Step 6: Automated EDA & Figures
    print("\n--- [STAGE 6/10] High-Resolution EDA & Visual Report Generation ---")
    from eda.run_eda import run_eda_visualizations
    run_eda_visualizations()

    # Step 7: MongoDB NoSQL Operations (PI 1.4.2.1)
    print("\n--- [STAGE 7/10] MongoDB NoSQL Database Engine ---")
    from mongodb.mongodb_operations import run_mongodb_operations
    run_mongodb_operations()

    # Step 8: HDFS Storage Experiments (PI 2.2.1.1)
    print("\n--- [STAGE 8/10] HDFS Distributed Storage Experiments ---")
    from hdfs.hdfs_experiments import run_hdfs_experiments
    run_hdfs_experiments()

    # Step 9: Apache Hive Analytics (PI 2.2.4.1)
    print("\n--- [STAGE 9/10] Apache Hive Analytical Engine ---")
    from hive.hive_analytics import run_hive_analytics
    run_hive_analytics()

    # Step 10: Streaming Pipeline (PI 3.2.2.1 + PI 3.3.2.1)
    print("\n--- [STAGE 10/10] Real-Time Streaming Pipeline & Window Analysis ---")
    from streaming.spark_streaming_pipeline import run_streaming_pipeline
    run_streaming_pipeline()

    total_elapsed = time.time() - total_start
    print("\n=========================================================================")
    print(f"[+] END-TO-END BIG DATA ANALYTICS PIPELINE COMPLETED IN {total_elapsed:.2f}s!")
    print("=========================================================================")
    print("\nProject Artifacts Created:")
    print("  * Raw Transaction Dataset:   data/raw/ecommerce_transactions.csv (~23.6 MB)")
    print("  * Processed Parquet Storage: data/processed/cleaned_transactions.parquet")
    print("  * MapReduce Output TSVs:     data/processed/mapreduce_*.tsv")
    print("  * Segmented Customers:       data/processed/rfm_segmented_customers.csv")
    print("  * Trained ML Models:         ml/model_artifacts/")
    print("  * High-Res Figures:          eda/figures/ (8 figures)")
    print("  * MongoDB Benchmarks:        data/processed/mongodb_benchmark_results.csv")
    print("  * HDFS Experiments:          data/processed/hdfs_experiment_results.csv")
    print("  * Hive Benchmarks:           data/processed/hive_benchmark_results.csv")
    print("  * Hive vs MongoDB:           data/processed/hive_vs_mongodb_comparison.csv")
    print("  * Streaming Results:         data/processed/streaming_*_window_results.csv")
    print("  * Window Comparison:         data/processed/streaming_window_comparison.csv")
    print("\nTo launch the interactive Big Data Web Dashboard:")
    print("  streamlit run dashboard/app.py")
    print("=========================================================================\n")

if __name__ == "__main__":
    run_master_pipeline()

