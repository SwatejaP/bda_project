"""
OmniPulse Big Data Analytics: Real-Time Stream Data Producer
Simulates a real-time e-commerce transaction stream by reading from the dataset
and emitting records over a TCP socket (for Spark Structured Streaming)
or writing to a streaming directory (file-based streaming source).

Usage:
  # Socket mode (default): Streams data over TCP port 9999
  python stream_producer.py --mode socket --port 9999

  # File mode: Writes micro-batches to a streaming directory
  python stream_producer.py --mode file --output-dir ../data/streaming/
"""

import os
import sys
import time
import json
import socket
import random
import argparse
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def load_source_data():
    """Load the source CSV dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, "data", "raw", "ecommerce_transactions.csv")

    if not os.path.exists(raw_csv):
        print(f"[!] Dataset not found at {raw_csv}. Please run data generation first.")
        sys.exit(1)

    df = pd.read_csv(raw_csv)
    print(f"[+] Loaded {len(df):,} source records for stream simulation.")
    return df


def record_to_json(row, event_time=None):
    """Convert a DataFrame row to a JSON string with an event timestamp."""
    if event_time is None:
        event_time = datetime.now()

    record = {
        "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_id": row.get("Transaction_ID", ""),
        "customer_id": row.get("Customer_ID", ""),
        "product_category": row.get("Product_Category", ""),
        "sub_category": row.get("Sub_Category", ""),
        "unit_price": float(row.get("Unit_Price", 0)),
        "quantity": int(row.get("Quantity", 0)),
        "total_amount": float(row.get("Total_Amount", 0)),
        "payment_method": row.get("Payment_Method", ""),
        "customer_region": row.get("Customer_Region", ""),
        "country": row.get("Country", ""),
        "is_fraudulent": int(row.get("Is_Fraudulent", 0)),
        "device_channel": row.get("Device_Channel", ""),
        "delivery_status": row.get("Delivery_Status", ""),
        "customer_age": int(row.get("Customer_Age", 0)),
        "discount_percent": float(row.get("Discount_Percent", 0)),
        "shipping_cost": float(row.get("Shipping_Cost", 0))
    }
    return json.dumps(record)


# ─────────────────────────────────────────────────────────
# Socket-Based Stream Producer
# ─────────────────────────────────────────────────────────

def run_socket_producer(host="localhost", port=9999, records_per_second=50, duration_seconds=120):
    """Stream records over a TCP socket for Spark Structured Streaming."""
    print(f"\n[*] Starting Socket Stream Producer on {host}:{port}")
    print(f"    Rate: {records_per_second} records/sec | Duration: {duration_seconds}s")
    print(f"    Total expected records: ~{records_per_second * duration_seconds:,}")

    df = load_source_data()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(60)

    print(f"[*] Waiting for Spark Streaming consumer to connect on port {port}...")

    try:
        conn, addr = server.accept()
        print(f"[+] Consumer connected from {addr}")

        total_sent = 0
        start_time = time.time()
        current_event_time = datetime.now()

        while time.time() - start_time < duration_seconds:
            batch_start = time.time()

            for _ in range(records_per_second):
                # Sample a random row from the dataset
                idx = random.randint(0, len(df) - 1)
                row = df.iloc[idx]

                # Advance event time with slight jitter
                current_event_time += timedelta(milliseconds=random.randint(10, 50))
                json_record = record_to_json(row, current_event_time)

                try:
                    conn.sendall((json_record + "\n").encode("utf-8"))
                    total_sent += 1
                except BrokenPipeError:
                    print("[!] Consumer disconnected.")
                    return total_sent

            elapsed_in_batch = time.time() - batch_start
            sleep_time = max(0, 1.0 - elapsed_in_batch)
            if sleep_time > 0:
                time.sleep(sleep_time)

            if total_sent % (records_per_second * 10) == 0:
                elapsed = time.time() - start_time
                print(f"    [{elapsed:.0f}s] Sent {total_sent:,} records "
                      f"(throughput: {total_sent / elapsed:.0f} rec/s)")

        elapsed = time.time() - start_time
        print(f"\n[+] Stream completed: {total_sent:,} records in {elapsed:.1f}s "
              f"({total_sent / elapsed:.0f} rec/s)")

        conn.close()
        return total_sent

    except socket.timeout:
        print("[!] No consumer connected within 60s timeout.")
        return 0
    finally:
        server.close()


# ─────────────────────────────────────────────────────────
# File-Based Stream Producer (for Structured Streaming)
# ─────────────────────────────────────────────────────────

def run_file_producer(output_dir, records_per_batch=100, batch_interval_seconds=2, num_batches=60):
    """Write micro-batch JSON files for Spark Structured Streaming file source."""
    print(f"\n[*] Starting File-Based Stream Producer")
    print(f"    Output: {output_dir}")
    print(f"    Batch size: {records_per_batch} records | Interval: {batch_interval_seconds}s | Batches: {num_batches}")

    os.makedirs(output_dir, exist_ok=True)

    df = load_source_data()
    total_sent = 0
    start_time = time.time()
    current_event_time = datetime.now()

    for batch_num in range(1, num_batches + 1):
        batch_records = []

        for _ in range(records_per_batch):
            idx = random.randint(0, len(df) - 1)
            row = df.iloc[idx]
            current_event_time += timedelta(milliseconds=random.randint(10, 50))
            batch_records.append(record_to_json(row, current_event_time))

        # Write batch file
        batch_filename = f"batch_{batch_num:04d}_{int(time.time())}.json"
        batch_path = os.path.join(output_dir, batch_filename)

        with open(batch_path, "w", encoding="utf-8") as f:
            f.write("\n".join(batch_records))

        total_sent += records_per_batch

        if batch_num % 10 == 0:
            elapsed = time.time() - start_time
            print(f"    [Batch {batch_num}/{num_batches}] Written {total_sent:,} records "
                  f"({elapsed:.1f}s elapsed)")

        time.sleep(batch_interval_seconds)

    elapsed = time.time() - start_time
    print(f"\n[+] File stream completed: {total_sent:,} records in {num_batches} batches "
          f"({elapsed:.1f}s)")

    return total_sent


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def start_producer(mode="file", port=9999, output_dir=None, duration=120, rate=50, batch_size=100, num_batches=60):
    """Start the stream producer in the specified mode."""
    if mode == "socket":
        return run_socket_producer(
            port=port,
            records_per_second=rate,
            duration_seconds=duration
        )
    else:
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "data", "streaming")
        return run_file_producer(
            output_dir=output_dir,
            records_per_batch=batch_size,
            num_batches=num_batches
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniPulse Stream Data Producer")
    parser.add_argument("--mode", choices=["socket", "file"], default="file",
                        help="Streaming mode: 'socket' for TCP or 'file' for file-based")
    parser.add_argument("--port", type=int, default=9999, help="TCP port for socket mode")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir for file mode")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds (socket mode)")
    parser.add_argument("--rate", type=int, default=50, help="Records per second (socket mode)")
    parser.add_argument("--batch-size", type=int, default=100, help="Records per batch (file mode)")
    parser.add_argument("--num-batches", type=int, default=60, help="Number of batches (file mode)")

    args = parser.parse_args()
    start_producer(
        mode=args.mode,
        port=args.port,
        output_dir=args.output_dir,
        duration=args.duration,
        rate=args.rate,
        batch_size=args.batch_size,
        num_batches=args.num_batches
    )
