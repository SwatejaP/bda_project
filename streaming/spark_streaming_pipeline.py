"""
OmniPulse Big Data Analytics: Spark Structured Streaming Pipeline (Kaggle Schema)
Reads a continuous stream of Kaggle eCommerce events, applies tumbling and sliding windows,
and aggregates live revenue and interaction counts by category and brand.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, count, sum, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, LongType
import os

def run_streaming_pipeline():
    print("[*] Initializing Spark Structured Streaming Engine...")
    
    spark = SparkSession.builder \
        .appName("OmniPulse_Kaggle_Stream") \
        .config("spark.sql.shuffle.partitions", "2") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Kaggle Schema Definition
    schema = StructType([
        StructField("event_time", TimestampType(), True),
        StructField("event_type", StringType(), True),
        StructField("product_id", LongType(), True),
        StructField("category_id", LongType(), True),
        StructField("category_code", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("user_id", LongType(), True),
        StructField("user_session", StringType(), True)
    ])

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    
    # We simulate streaming by reading CSVs from the raw data directory as a stream
    raw_stream = spark.readStream \
        .schema(schema) \
        .option("header", "true") \
        .option("maxFilesPerTrigger", 1) \
        .csv(raw_data_dir)

    print("[+] Stream schema initialized. Processing windowed aggregations...")

    # 1. Clean data
    clean_stream = raw_stream.na.fill({"category_code": "unknown", "brand": "unknown", "price": 0.0})

    # 2. Tumbling Window: Total Purchases and Revenue by Category (every 1 hour of event_time)
    tumbling_window_agg = clean_stream \
        .filter(col("event_type") == "purchase") \
        .withWatermark("event_time", "2 hours") \
        .groupBy(
            window(col("event_time"), "1 hour"),
            col("category_code")
        ) \
        .agg(
            sum("price").alias("total_revenue"),
            count("product_id").alias("purchase_count")
        )

    # 3. Sliding Window: Active Event Interactions by Brand (1 hour window, sliding every 30 mins)
    sliding_window_agg = clean_stream \
        .withWatermark("event_time", "2 hours") \
        .groupBy(
            window(col("event_time"), "1 hour", "30 minutes"),
            col("brand")
        ) \
        .agg(
            count("product_id").alias("total_interactions"),
            sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchases")
        )

    # Output to Console Sink
    query1 = tumbling_window_agg.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    query2 = sliding_window_agg.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    print("[*] Streaming pipeline active. Awaiting data...")
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    run_streaming_pipeline()
