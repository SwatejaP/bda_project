# """
# OmniPulse Big Data Analytics: Data Ingestion & Enrichment
# Reads from the Kaggle eCommerce dataset and enriches it with enterprise dimensions.
# """

# import os
# import sys
# import pandas as pd
# import numpy as np

# def generate_ecommerce_dataset(num_records: int = 125000, output_path: str = None) -> str:
#     print(f"[*] Extracting {num_records:,} records from local dataset...")

#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     dataset_dir = os.path.join(base_dir, "dataset")
#     source_file = os.path.join(dataset_dir, "2019-Oct.csv")

#     if output_path is None:
#         output_dir = os.path.join(base_dir, "data", "raw")
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, "ecommerce_transactions.csv")
#     else:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     if not os.path.exists(source_file):
#         print(f"[!] Source dataset not found: {source_file}")
#         return output_path

#     # Read the required number of records
#     df = pd.read_csv(source_file, nrows=num_records)
    
#     print("[*] Enriching Kaggle dataset with Enterprise Schema columns...")
#     np.random.seed(42)
#     n = len(df)
    
#     # Map existing Kaggle columns to Enterprise schema where possible
#     df["Transaction_ID"] = df["user_session"]
#     df["Customer_ID"] = "CUST_" + df["user_id"].astype(str)
#     df["Order_Date"] = df["event_time"]
#     df["Customer_Age"] = np.random.randint(18, 65, size=n)
#     df["Customer_Gender"] = np.random.choice(["Male", "Female"], size=n)
#     df["Customer_Region"] = np.random.choice(["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"], size=n)
#     df["Country"] = np.random.choice(["USA", "Canada", "UK", "Germany", "France", "Japan", "Australia", "India", "Brazil"], size=n)
#     df["Product_Category"] = df["category_code"].fillna("general")
#     df["Sub_Category"] = df["brand"].fillna("unbranded")
#     df["Product_ID"] = df["product_id"]
#     df["Unit_Price"] = df["price"]
#     df["Quantity"] = 1
#     df["Discount_Percent"] = np.round(np.random.uniform(0.0, 0.2, size=n), 2)
#     df["Shipping_Cost"] = np.round(np.random.uniform(5.0, 25.0, size=n), 2)
#     df["Total_Amount"] = df["price"]
#     df["Payment_Method"] = np.random.choice(["Credit Card", "PayPal", "Debit Card", "UPI / Digital Wallet", "Cash on Delivery"], size=n)
#     df["Device_Channel"] = np.random.choice(["Mobile App (Android)", "Mobile App (iOS)", "Web Desktop", "Mobile Web", "Tablet"], size=n)
#     df["Order_Priority"] = np.random.choice(["Standard", "Express Same-Day", "Next-Day"], size=n, p=[0.7, 0.2, 0.1])
#     df["Delivery_Status"] = np.random.choice(["Delivered", "Shipped", "Processing", "Returned"], size=n, p=[0.7, 0.15, 0.1, 0.05])
#     df["Customer_Rating"] = np.round(np.random.uniform(1.0, 5.0, size=n), 1)
#     df["Session_Duration_Min"] = np.round(np.random.uniform(1.0, 45.0, size=n), 1)
#     df["Is_Fraudulent"] = np.random.choice([0, 1], size=n, p=[0.98, 0.02])

#     df.to_csv(output_path, index=False)

#     file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
#     print(f"[+] Dataset successfully ingested and enriched at: {output_path}")
#     print(f"[+] Total Records: {len(df):,}")
#     print(f"[+] File Size: {file_size_mb:.2f} MB")
    
#     return output_path

# if __name__ == "__main__":
#     records = 125000
#     if len(sys.argv) > 1:
#         records = int(sys.argv[1])
#     generate_ecommerce_dataset(num_records=records)

"""
OmniPulse Big Data Analytics: Data Ingestion & Enrichment
Reads October + November Kaggle eCommerce datasets and enriches
them with enterprise dimensions using chunked processing for
large-scale (multi-GB) output.
"""

import os
import sys
import time
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# ENTERPRISE ENRICHMENT  (applied per chunk)
# ─────────────────────────────────────────────────────────────

def _enrich_chunk(df: pd.DataFrame, chunk_seed: int) -> pd.DataFrame:
    """Add all enterprise columns to a single chunk."""

    rng = np.random.default_rng(chunk_seed)
    n = len(df)

    # Customer Dimensions
    df["Transaction_ID"]    = df["user_session"]
    df["Customer_ID"]       = "CUST_" + df["user_id"].astype(str)
    df["Customer_Age"]      = rng.integers(18, 65, size=n)
    df["Customer_Gender"]   = rng.choice(["Male", "Female"], size=n)
    df["Customer_Region"]   = rng.choice(
        ["North America", "Europe", "Asia-Pacific",
         "Latin America", "Middle East & Africa"],
        size=n
    )
    df["Country"] = rng.choice(
        ["USA", "Canada", "UK", "Germany", "France",
         "Japan", "Australia", "India", "Brazil"],
        size=n
    )

    # Product Dimensions
    df["Product_Category"] = df["category_code"].fillna("general")
    df["Sub_Category"]     = df["brand"].fillna("unbranded")
    df["Product_ID"]       = df["product_id"]

    # Transaction Dimensions
    df["Order_Date"]       = df["event_time"]
    df["Unit_Price"]       = df["price"]
    df["Quantity"]         = 1
    df["Discount_Percent"] = np.round(rng.uniform(0.0, 0.2, size=n), 2)
    df["Shipping_Cost"]    = np.round(rng.uniform(5.0, 25.0, size=n), 2)
    df["Total_Amount"]     = df["price"]

    # Payment / Channel
    df["Payment_Method"] = rng.choice(
        ["Credit Card", "PayPal", "Debit Card",
         "UPI / Digital Wallet", "Cash on Delivery"],
        size=n
    )
    df["Device_Channel"] = rng.choice(
        ["Mobile App (Android)", "Mobile App (iOS)",
         "Web Desktop", "Mobile Web", "Tablet"],
        size=n
    )

    # Order / Delivery
    df["Order_Priority"] = rng.choice(
        ["Standard", "Express Same-Day", "Next-Day"],
        size=n,
        p=[0.7, 0.2, 0.1]
    )
    df["Delivery_Status"] = rng.choice(
        ["Delivered", "Shipped", "Processing", "Returned"],
        size=n,
        p=[0.70, 0.15, 0.10, 0.05]
    )

    # Customer Behaviour
    df["Customer_Rating"]      = np.round(rng.uniform(1.0, 5.0, size=n), 1)
    df["Session_Duration_Min"] = np.round(rng.uniform(1.0, 45.0, size=n), 1)

    # Fraud — only purchase rows can be fraudulent
    is_purchase = (df["event_type"] == "purchase").to_numpy()
    fraud = np.zeros(n, dtype=np.int8)
    n_purchases = int(is_purchase.sum())
    if n_purchases > 0:
        fraud[is_purchase] = rng.choice(
            [0, 1], size=n_purchases, p=[0.98, 0.02]
        )
    df["Is_Fraudulent"] = fraud

    return df


# ─────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_ecommerce_dataset(
    target_gb: float = 4.0,
    chunk_size: int = 200_000,
    output_path: str = None
) -> str:
    """
    Stream-reads the Kaggle CSV files in chunks, enriches each
    chunk, and appends to a single output CSV until the file
    reaches *target_gb* gigabytes on disk.

    Parameters
    ----------
    target_gb   : Target output file size in GB (default 4.0)
    chunk_size  : Rows processed per iteration (default 200 000)
    output_path : Override output location (optional)
    """

    target_bytes = int(target_gb * 1024 ** 3)

    # ── locate files ──────────────────────────────────────────
    base_dir      = os.path.dirname(os.path.abspath(__file__))
    dataset_dir   = os.path.join(base_dir, "dataset")
    october_file  = os.path.join(dataset_dir, "2019-Oct.csv")
    november_file = os.path.join(dataset_dir, "2019-Nov.csv")

    for label, path in [("October", october_file), ("November", november_file)]:
        if not os.path.exists(path):
            print(f"[!] {label} dataset not found: {path}")
            sys.exit(1)

    # ── output path ───────────────────────────────────────────
    if output_path is None:
        output_dir  = os.path.join(base_dir, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ecommerce_transactions.csv")
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # Remove existing file so we start fresh
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"[!] Removed existing file: {output_path}")

    print("=" * 62)
    print("  OmniPulse  –  Large-Scale Data Ingestion")
    print("=" * 62)
    print(f"  Target size  : {target_gb:.1f} GB  ({target_bytes:,} bytes)")
    print(f"  Chunk size   : {chunk_size:,} rows per iteration")
    print(f"  Sources      : Oct 2019  +  Nov 2019")
    print(f"  Output       : {output_path}")
    print("=" * 62)

    # ── streaming loop ────────────────────────────────────────
    total_rows   = 0
    chunk_index  = 0
    header_done  = False
    start_time   = time.time()
    sources      = [october_file, november_file]

    # Infinite round-robin: restart each file from scratch when exhausted
    def _infinite_chunks(filepaths, csize):
        idx = 0
        while True:
            path = filepaths[idx % len(filepaths)]
            for chunk in pd.read_csv(path, chunksize=csize, low_memory=False):
                yield chunk
            idx += 1  # move to next source; wraps via modulo

    chunk_iter = _infinite_chunks(sources, chunk_size)

    while True:
        # Check target size
        current_bytes = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        if current_bytes >= target_bytes:
            break

        # Fetch next chunk from infinite round-robin
        chunk = next(chunk_iter)

        # Enrich and write
        chunk = _enrich_chunk(chunk.copy(), chunk_seed=chunk_index * 7919)
        chunk.to_csv(output_path, mode="a", header=not header_done, index=False)
        header_done  = True
        total_rows  += len(chunk)
        chunk_index += 1

        # Progress every 10 chunks
        if chunk_index % 10 == 0:
            elapsed      = time.time() - start_time
            size_gb      = current_bytes / 1024 ** 3
            pct          = current_bytes / target_bytes * 100
            speed_k      = total_rows / elapsed / 1000 if elapsed else 0
            eta_sec      = ((target_bytes - current_bytes) / (current_bytes / elapsed)
                            if elapsed and current_bytes else 0)
            print(
                f"  [{chunk_index:>5} chunks]  "
                f"rows={total_rows:>12,}  "
                f"size={size_gb:>5.2f} GB / {target_gb:.1f} GB  "
                f"({pct:>5.1f}%)  "
                f"speed={speed_k:>5.1f}k rows/s  "
                f"ETA={eta_sec/60:.1f}min"
            )

    # ── final stats ───────────────────────────────────────────
    elapsed     = time.time() - start_time
    final_bytes = os.path.getsize(output_path)
    final_gb    = final_bytes / 1024 ** 3

    print()
    print("=" * 62)
    print("  DATA INGESTION COMPLETED")
    print("=" * 62)
    print(f"  Total rows written  : {total_rows:,}")
    print(f"  Final file size     : {final_gb:.3f} GB  ({final_bytes:,} bytes)")
    print(f"  Columns per row     : 31")
    print(f"  Time elapsed        : {elapsed/60:.1f} minutes")
    print(f"  Output              : {output_path}")
    print("=" * 62)

    return output_path


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    target_gb  = 4.0
    chunk_size = 50_000

    if len(sys.argv) > 1:
        target_gb = float(sys.argv[1])
    if len(sys.argv) > 2:
        chunk_size = int(sys.argv[2])

    generate_ecommerce_dataset(
        target_gb  = target_gb,
        chunk_size = chunk_size,
    )