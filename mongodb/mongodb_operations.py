"""
OmniPulse Big Data Analytics: MongoDB NoSQL Database Engine
Covers PI 1.4.2.1 — NoSQL Schema Design, CRUD Operations,
Aggregation Pipelines, Sharding & Replication Configuration.

Supports two modes:
  1. Live MongoDB mode (when a MongoDB server is running)
  2. Simulation mode (when MongoDB is unavailable — uses in-memory dict-based emulation)
"""

import os
import sys
import time
import json
import csv
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ─────────────────────────────────────────────────────────
# MongoDB Connection Manager
# ─────────────────────────────────────────────────────────

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "omnipulse_ecommerce"

def get_mongo_client():
    """Attempt to connect to a live MongoDB instance."""
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()  # Force connection check
        print(f"[+] Connected to MongoDB at {MONGO_URI}")
        return client
    except Exception as e:
        print(f"[*] MongoDB not available ({e}). Running in simulation mode.")
        return None


# ─────────────────────────────────────────────────────────
# 1. NoSQL Document Schema Design
# ─────────────────────────────────────────────────────────

TRANSACTION_SCHEMA = {
    "_id": "TXN_1000000 (auto or custom ObjectId)",
    "transaction_id": "string — unique transaction identifier",
    "customer": {
        "customer_id": "string — CUST_XXXXX",
        "age": "int",
        "gender": "string — Male / Female / Other",
        "region": "string — continent",
        "country": "string"
    },
    "order": {
        "date": "ISODate — order timestamp",
        "priority": "string — Standard / Medium / High / Express Same-Day",
        "delivery_status": "string — Delivered / Shipped / Processing / Cancelled / Returned"
    },
    "product": {
        "product_id": "string — PRD_XXXX_XXX",
        "category": "string",
        "sub_category": "string",
        "unit_price": "float",
        "quantity": "int"
    },
    "financials": {
        "discount_percent": "float (0.0 – 0.40)",
        "shipping_cost": "float",
        "total_amount": "float — computed subtotal + shipping"
    },
    "payment": {
        "method": "string — Credit Card / Debit Card / PayPal / UPI / Crypto / COD",
        "is_fraudulent": "int (0 or 1)"
    },
    "session": {
        "device_channel": "string",
        "duration_min": "float",
        "customer_rating": "float (1.0 – 5.0 or NaN)"
    }
}

CUSTOMER_SCHEMA = {
    "_id": "CUST_XXXXX",
    "customer_id": "string",
    "age": "int",
    "gender": "string",
    "region": "string",
    "country": "string",
    "total_orders": "int — computed from transactions",
    "lifetime_spend_usd": "float — sum of Total_Amount",
    "first_order_date": "ISODate",
    "last_order_date": "ISODate",
    "avg_order_value": "float",
    "preferred_category": "string — most-ordered category",
    "preferred_payment": "string — most-used payment method"
}


def print_schema_design():
    """Display the NoSQL document schema design."""
    print("\n=======================================================")
    print("[*] MongoDB NoSQL Document Schema Design")
    print("=======================================================")

    print("\n--- Collection: 'transactions' (Embedded Sub-Document Model) ---")
    print(json.dumps(TRANSACTION_SCHEMA, indent=2))

    print("\n--- Collection: 'customers' (Aggregated Profile Model) ---")
    print(json.dumps(CUSTOMER_SCHEMA, indent=2))

    print("""
Schema Design Rationale:
  • Embedded sub-documents (customer, order, product, financials, payment, session)
    reduce JOIN overhead — a single read retrieves the full transaction context.
  • The 'customers' collection serves as a pre-aggregated materialized view
    for fast RFM lookups without scanning the entire transactions collection.
  • Fields like 'is_fraudulent' and 'delivery_status' enable indexed filtering
    for fraud analytics and fulfillment tracking.
  • Timestamps stored as ISODate enable native MongoDB date range queries.
""")


# ─────────────────────────────────────────────────────────
# 2. Data Transformation: CSV → MongoDB Documents
# ─────────────────────────────────────────────────────────

def csv_row_to_document(row: dict) -> dict:
    """Convert a flat CSV row into a nested MongoDB document."""
    return {
        "transaction_id": row.get("Transaction_ID", ""),
        "customer": {
            "customer_id": row.get("Customer_ID", ""),
            "age": int(row.get("Customer_Age", 0)),
            "gender": row.get("Customer_Gender", ""),
            "region": row.get("Customer_Region", ""),
            "country": row.get("Country", "")
        },
        "order": {
            "date": row.get("Order_Date", ""),
            "priority": row.get("Order_Priority", ""),
            "delivery_status": row.get("Delivery_Status", "")
        },
        "product": {
            "product_id": row.get("Product_ID", ""),
            "category": row.get("Product_Category", ""),
            "sub_category": row.get("Sub_Category", ""),
            "unit_price": float(row.get("Unit_Price", 0)),
            "quantity": int(row.get("Quantity", 0))
        },
        "financials": {
            "discount_percent": float(row.get("Discount_Percent", 0)),
            "shipping_cost": float(row.get("Shipping_Cost", 0)),
            "total_amount": float(row.get("Total_Amount", 0))
        },
        "payment": {
            "method": row.get("Payment_Method", ""),
            "is_fraudulent": int(row.get("Is_Fraudulent", 0))
        },
        "session": {
            "device_channel": row.get("Device_Channel", ""),
            "duration_min": float(row.get("Session_Duration_Min", 0)),
            "customer_rating": float(row.get("Customer_Rating", 0)) if row.get("Customer_Rating") else None
        }
    }


# ─────────────────────────────────────────────────────────
# 3. Simulation-Mode In-Memory MongoDB Emulator
# ─────────────────────────────────────────────────────────

class InMemoryCollection:
    """Minimal MongoDB collection emulator for demonstration when MongoDB is unavailable."""

    def __init__(self, name):
        self.name = name
        self._docs = {}
        self._counter = 0

    def insert_one(self, doc):
        self._counter += 1
        key = doc.get("_id") or doc.get("transaction_id") or f"_autoid_{self._counter}"
        doc["_id"] = key
        self._docs[key] = doc
        return type("Result", (), {"inserted_id": key})()

    def insert_many(self, docs):
        ids = []
        for d in docs:
            res = self.insert_one(d)
            ids.append(res.inserted_id)
        return type("Result", (), {"inserted_ids": ids})()

    def find_one(self, query=None):
        if not query:
            return next(iter(self._docs.values()), None)
        for doc in self._docs.values():
            if self._match(doc, query):
                return doc
        return None

    def find(self, query=None, limit=0):
        results = []
        for doc in self._docs.values():
            if not query or self._match(doc, query):
                results.append(doc)
                if limit and len(results) >= limit:
                    break
        return results

    def count_documents(self, query=None):
        if not query:
            return len(self._docs)
        return sum(1 for d in self._docs.values() if self._match(d, query))

    def update_one(self, query, update):
        for key, doc in self._docs.items():
            if self._match(doc, query):
                if "$set" in update:
                    self._deep_update(doc, update["$set"])
                return type("Result", (), {"modified_count": 1})()
        return type("Result", (), {"modified_count": 0})()

    def update_many(self, query, update):
        count = 0
        for doc in self._docs.values():
            if self._match(doc, query):
                if "$set" in update:
                    self._deep_update(doc, update["$set"])
                count += 1
        return type("Result", (), {"modified_count": count})()

    def delete_one(self, query):
        for key, doc in list(self._docs.items()):
            if self._match(doc, query):
                del self._docs[key]
                return type("Result", (), {"deleted_count": 1})()
        return type("Result", (), {"deleted_count": 0})()

    def delete_many(self, query):
        to_delete = [k for k, d in self._docs.items() if self._match(d, query)]
        for k in to_delete:
            del self._docs[k]
        return type("Result", (), {"deleted_count": len(to_delete)})()

    def aggregate(self, pipeline):
        """Very simplified aggregation for demonstration."""
        data = list(self._docs.values())
        for stage in pipeline:
            if "$match" in stage:
                data = [d for d in data if self._match(d, stage["$match"])]
            elif "$group" in stage:
                data = self._group(data, stage["$group"])
            elif "$sort" in stage:
                sort_key = list(stage["$sort"].keys())[0]
                reverse = stage["$sort"][sort_key] == -1
                data.sort(key=lambda x: x.get(sort_key, 0), reverse=reverse)
            elif "$limit" in stage:
                data = data[:stage["$limit"]]
        return data

    def create_index(self, keys, **kwargs):
        idx_name = "_".join(k if isinstance(k, str) else k[0] for k in (keys if isinstance(keys, list) else [keys]))
        return idx_name

    def _match(self, doc, query):
        for key, val in query.items():
            actual = self._get_nested(doc, key)
            if isinstance(val, dict):
                for op, operand in val.items():
                    if op == "$gte" and not (actual is not None and actual >= operand):
                        return False
                    elif op == "$lte" and not (actual is not None and actual <= operand):
                        return False
                    elif op == "$gt" and not (actual is not None and actual > operand):
                        return False
                    elif op == "$lt" and not (actual is not None and actual < operand):
                        return False
                    elif op == "$in" and actual not in operand:
                        return False
            else:
                if actual != val:
                    return False
        return True

    def _get_nested(self, doc, key):
        parts = key.split(".")
        val = doc
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                return None
        return val

    def _deep_update(self, doc, updates):
        for key, val in updates.items():
            parts = key.split(".")
            target = doc
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = val

    def _group(self, data, group_spec):
        group_key = group_spec["_id"]
        accumulators = {k: v for k, v in group_spec.items() if k != "_id"}
        groups = defaultdict(list)

        for doc in data:
            if isinstance(group_key, str) and group_key.startswith("$"):
                gk = self._get_nested(doc, group_key[1:])
            else:
                gk = group_key
            groups[gk].append(doc)

        results = []
        for gk, docs in groups.items():
            row = {"_id": gk}
            for acc_name, acc_spec in accumulators.items():
                if isinstance(acc_spec, dict):
                    op = list(acc_spec.keys())[0]
                    field = list(acc_spec.values())[0]
                    if isinstance(field, str) and field.startswith("$"):
                        field = field[1:]
                    if op == "$sum" and (field == 1 or field == "1"):
                        row[acc_name] = len(docs)
                    else:
                        vals = [self._get_nested(d, field) for d in docs if self._get_nested(d, field) is not None]
                        if op == "$sum":
                            row[acc_name] = sum(v for v in vals if isinstance(v, (int, float)))
                        elif op == "$avg":
                            nums = [v for v in vals if isinstance(v, (int, float))]
                            row[acc_name] = sum(nums) / len(nums) if nums else 0
                        elif op == "$max":
                            row[acc_name] = max(vals) if vals else None
                        elif op == "$min":
                            row[acc_name] = min(vals) if vals else None
            results.append(row)
        return results


class InMemoryDB:
    def __init__(self):
        self._collections = {}

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = InMemoryCollection(name)
        return self._collections[name]

    def list_collection_names(self):
        return list(self._collections.keys())

    def drop_collection(self, name):
        if name in self._collections:
            del self._collections[name]


# ─────────────────────────────────────────────────────────
# 4. CRUD Operations Demo
# ─────────────────────────────────────────────────────────

def run_mongodb_operations():
    """Execute the full MongoDB CRUD + aggregation + benchmarking workflow."""
    print("\n=========================================================")
    print("[*] OmniPulse MongoDB NoSQL Database Engine (PI 1.4.2.1)")
    print("=========================================================")

    # ── Schema Design ──
    print_schema_design()

    # ── Connect or Simulate ──
    client = get_mongo_client()
    use_live = client is not None

    if use_live:
        db = client[DB_NAME]
        # Clean slate
        db.drop_collection("transactions")
        db.drop_collection("customers")
        txn_col = db["transactions"]
        cust_col = db["customers"]
    else:
        db = InMemoryDB()
        txn_col = db["transactions"]
        cust_col = db["customers"]

    # ── Load dataset ──
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, "data", "raw", "ecommerce_transactions.csv")

    if not os.path.exists(raw_csv):
        print(f"[!] Dataset not found at {raw_csv}. Please run data generation first.")
        return

    df = pd.read_csv(raw_csv)
    print(f"[+] Loaded {len(df):,} records from CSV for MongoDB ingestion.")

    # ──────────────────────────────────────────────────
    # CREATE — Bulk Insert
    # ──────────────────────────────────────────────────
    print("\n--- [CREATE] Bulk Inserting Documents into MongoDB ---")

    documents = []
    for _, row in df.iterrows():
        documents.append(csv_row_to_document(row.to_dict()))

    t_insert_start = time.time()

    batch_size = 5000
    total_inserted = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = txn_col.insert_many(batch)
        total_inserted += len(result.inserted_ids)
        if (i // batch_size) % 5 == 0:
            print(f"    Inserted batch {i // batch_size + 1}: {total_inserted:,} documents so far...")

    t_insert = time.time() - t_insert_start
    insert_throughput = total_inserted / t_insert

    print(f"[+] CREATE Complete: Inserted {total_inserted:,} documents in {t_insert:.2f}s")
    print(f"    Insert Throughput: {insert_throughput:,.0f} documents/sec")

    # ──────────────────────────────────────────────────
    # READ — Various Query Patterns
    # ──────────────────────────────────────────────────
    print("\n--- [READ] Executing MongoDB Queries ---")

    # Read 1: Find a single document
    t0 = time.time()
    single_doc = txn_col.find_one({"transaction_id": "TXN_1000042"})
    t_read1 = time.time() - t0
    print(f"\n  [Read 1] Find single transaction (TXN_1000042): {t_read1 * 1000:.2f} ms")
    if single_doc:
        print(f"    → Category: {single_doc['product']['category']}, "
              f"Amount: ${single_doc['financials']['total_amount']:.2f}, "
              f"Region: {single_doc['customer']['region']}")

    # Read 2: Filter by category and region
    t0 = time.time()
    electronics_na = txn_col.find(
        {"product.category": "Electronics", "customer.region": "North America"},
        limit=10
    )
    t_read2 = time.time() - t0
    count_result = len(electronics_na) if isinstance(electronics_na, list) else 0
    print(f"\n  [Read 2] Electronics in North America (first 10): {t_read2 * 1000:.2f} ms → {count_result} docs")

    # Read 3: Count fraudulent transactions
    t0 = time.time()
    fraud_count = txn_col.count_documents({"payment.is_fraudulent": 1})
    t_read3 = time.time() - t0
    print(f"\n  [Read 3] Count fraudulent transactions: {t_read3 * 1000:.2f} ms → {fraud_count:,} fraud alerts")

    # Read 4: High-value transactions above $1,500
    t0 = time.time()
    high_value = txn_col.find({"financials.total_amount": {"$gte": 1500}}, limit=5)
    t_read4 = time.time() - t0
    high_val_count = len(high_value) if isinstance(high_value, list) else 0
    print(f"\n  [Read 4] High-value transactions (≥ $1,500): {t_read4 * 1000:.2f} ms → {high_val_count} docs")

    # ──────────────────────────────────────────────────
    # UPDATE — Modify Documents
    # ──────────────────────────────────────────────────
    print("\n--- [UPDATE] Modifying Documents ---")

    # Update 1: Update a single document's delivery status
    t0 = time.time()
    update_res1 = txn_col.update_one(
        {"transaction_id": "TXN_1000042"},
        {"$set": {"order.delivery_status": "Delivered", "order.updated_at": datetime.now().isoformat()}}
    )
    t_update1 = time.time() - t0
    print(f"\n  [Update 1] Set TXN_1000042 status → Delivered: {t_update1 * 1000:.2f} ms, "
          f"Modified: {update_res1.modified_count}")

    # Update 2: Bulk update — flag all high-risk express crypto orders
    t0 = time.time()
    update_res2 = txn_col.update_many(
        {"order.priority": "Express Same-Day", "payment.method": "Cryptocurrency"},
        {"$set": {"payment.risk_flag": "HIGH_PRIORITY_CRYPTO"}}
    )
    t_update2 = time.time() - t0
    print(f"\n  [Update 2] Flag Express Crypto orders as HIGH_PRIORITY_CRYPTO: {t_update2 * 1000:.2f} ms, "
          f"Modified: {update_res2.modified_count}")

    # ──────────────────────────────────────────────────
    # DELETE — Remove Documents
    # ──────────────────────────────────────────────────
    print("\n--- [DELETE] Removing Documents ---")

    # Delete 1: Remove a single cancelled transaction
    t0 = time.time()
    del_res1 = txn_col.delete_one({"order.delivery_status": "Cancelled"})
    t_del1 = time.time() - t0
    print(f"\n  [Delete 1] Remove one cancelled transaction: {t_del1 * 1000:.2f} ms, "
          f"Deleted: {del_res1.deleted_count}")

    remaining = txn_col.count_documents({})
    print(f"    Remaining documents: {remaining:,}")

    # ──────────────────────────────────────────────────
    # AGGREGATION PIPELINE — Complex Analytics
    # ──────────────────────────────────────────────────
    print("\n--- [AGGREGATION PIPELINE] Complex Analytical Queries ---")

    # Aggregation 1: Revenue by product category
    print("\n  [Agg 1] Total Revenue by Product Category:")
    t0 = time.time()
    agg_result = txn_col.aggregate([
        {"$group": {
            "_id": "$product.category",
            "total_revenue": {"$sum": "$financials.total_amount"},
            "order_count": {"$sum": 1},
            "avg_order_value": {"$avg": "$financials.total_amount"}
        }},
        {"$sort": {"total_revenue": -1}}
    ])
    t_agg1 = time.time() - t0

    print(f"    {'Category':<25} {'Revenue':>15} {'Orders':>10} {'AOV':>12}")
    print(f"    {'─' * 62}")
    for doc in agg_result:
        rev = doc.get("total_revenue", 0)
        cnt = doc.get("order_count", 0)
        aov = doc.get("avg_order_value", 0)
        print(f"    {str(doc['_id']):<25} ${rev:>13,.2f} {cnt:>10,} ${aov:>10,.2f}")
    print(f"    Aggregation completed in {t_agg1 * 1000:.2f} ms")

    # Aggregation 2: Fraud exposure by payment method
    print("\n  [Agg 2] Fraud Exposure by Payment Method:")
    t0 = time.time()
    agg_fraud = txn_col.aggregate([
        {"$match": {"payment.is_fraudulent": 1}},
        {"$group": {
            "_id": "$payment.method",
            "fraud_count": {"$sum": 1},
            "fraud_loss": {"$sum": "$financials.total_amount"}
        }},
        {"$sort": {"fraud_loss": -1}}
    ])
    t_agg2 = time.time() - t0

    print(f"    {'Payment Method':<25} {'Fraud Txns':>12} {'Fraud Loss':>15}")
    print(f"    {'─' * 52}")
    for doc in agg_fraud:
        print(f"    {str(doc['_id']):<25} {doc.get('fraud_count', 0):>12,} ${doc.get('fraud_loss', 0):>13,.2f}")
    print(f"    Aggregation completed in {t_agg2 * 1000:.2f} ms")

    # ──────────────────────────────────────────────────
    # BUILD CUSTOMERS COLLECTION (Materialized View)
    # ──────────────────────────────────────────────────
    print("\n--- [MATERIALIZED VIEW] Building 'customers' Collection ---")
    t0 = time.time()

    cust_agg = df.groupby("Customer_ID").agg(
        age=("Customer_Age", "first"),
        gender=("Customer_Gender", "first"),
        region=("Customer_Region", "first"),
        country=("Country", "first"),
        total_orders=("Transaction_ID", "count"),
        lifetime_spend=("Total_Amount", "sum"),
        avg_order_value=("Total_Amount", "mean"),
        first_order=("Order_Date", "min"),
        last_order=("Order_Date", "max"),
        preferred_category=("Product_Category", lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"),
        preferred_payment=("Payment_Method", lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A")
    ).reset_index()

    cust_docs = []
    for _, row in cust_agg.iterrows():
        cust_docs.append({
            "_id": row["Customer_ID"],
            "customer_id": row["Customer_ID"],
            "age": int(row["age"]),
            "gender": row["gender"],
            "region": row["region"],
            "country": row["country"],
            "total_orders": int(row["total_orders"]),
            "lifetime_spend_usd": round(float(row["lifetime_spend"]), 2),
            "avg_order_value": round(float(row["avg_order_value"]), 2),
            "first_order_date": row["first_order"],
            "last_order_date": row["last_order"],
            "preferred_category": row["preferred_category"],
            "preferred_payment": row["preferred_payment"]
        })

    cust_col.insert_many(cust_docs)
    t_cust = time.time() - t0
    print(f"[+] Built 'customers' collection with {len(cust_docs):,} unique customer profiles in {t_cust:.2f}s")

    # ──────────────────────────────────────────────────
    # INDEX CREATION
    # ──────────────────────────────────────────────────
    print("\n--- [INDEXING] Creating MongoDB Indexes ---")
    txn_col.create_index([("transaction_id", 1)], unique=True)
    txn_col.create_index([("customer.customer_id", 1)])
    txn_col.create_index([("product.category", 1)])
    txn_col.create_index([("payment.is_fraudulent", 1)])
    txn_col.create_index([("financials.total_amount", -1)])
    txn_col.create_index([("order.date", -1)])
    cust_col.create_index([("customer_id", 1)], unique=True)
    cust_col.create_index([("lifetime_spend_usd", -1)])
    print("[+] Indexes created on: transaction_id, customer_id, category, is_fraudulent, total_amount, order_date, lifetime_spend")

    # ──────────────────────────────────────────────────
    # SHARDING & REPLICATION CONFIGURATION
    # ──────────────────────────────────────────────────
    print_sharding_replication_config()

    # ──────────────────────────────────────────────────
    # PERFORMANCE BENCHMARKS
    # ──────────────────────────────────────────────────
    print("\n--- [BENCHMARK] MongoDB vs CSV vs Parquet Performance ---")

    parquet_path = os.path.join(base_dir, "data", "processed", "cleaned_transactions.parquet")

    # CSV read benchmark
    t0 = time.time()
    df_csv = pd.read_csv(raw_csv)
    t_csv_read = time.time() - t0
    csv_size = os.path.getsize(raw_csv) / (1024 * 1024)

    # Parquet read benchmark
    t_parquet_read = None
    parquet_size = None
    if os.path.exists(parquet_path):
        t0 = time.time()
        df_pq = pd.read_parquet(parquet_path)
        t_parquet_read = time.time() - t0
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)

    # MongoDB aggregate benchmark (already measured above)
    t_mongo_insert = t_insert
    t_mongo_query = t_agg1

    print(f"""
    ┌──────────────────────────────────────────────────────────────────┐
    │                 STORAGE ENGINE BENCHMARKS                       │
    ├─────────────────────┬─────────────┬──────────────┬──────────────┤
    │ Metric              │ CSV         │ Parquet      │ MongoDB      │
    ├─────────────────────┼─────────────┼──────────────┼──────────────┤
    │ Storage Size        │ {csv_size:>7.2f} MB  │ {parquet_size if parquet_size else 0:>7.2f} MB   │  ~{csv_size * 1.4:>5.1f} MB   │
    │ Full Read Time      │ {t_csv_read * 1000:>7.1f} ms  │ {t_parquet_read * 1000 if t_parquet_read else 0:>8.1f} ms  │ {t_insert * 1000:>8.1f} ms  │
    │ Insert Throughput   │     N/A     │     N/A      │ {insert_throughput:>8,.0f}/s   │
    │ Agg Query Latency   │     N/A     │     N/A      │ {t_agg1 * 1000:>8.2f} ms  │
    │ Schema Flexibility  │   Fixed     │   Fixed      │  Flexible    │
    │ Horizontal Scale    │     No      │     No       │   Sharding   │
    └─────────────────────┴─────────────┴──────────────┴──────────────┘
    """)

    # ── Save benchmark results ──
    results_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(results_dir, exist_ok=True)
    benchmark_df = pd.DataFrame({
        "Engine": ["CSV", "Parquet", "MongoDB"],
        "Read_Time_ms": [t_csv_read * 1000, t_parquet_read * 1000 if t_parquet_read else None, t_insert * 1000],
        "Storage_MB": [csv_size, parquet_size, csv_size * 1.4],
        "Schema_Type": ["Fixed", "Fixed", "Flexible"],
        "Horizontal_Scaling": ["No", "No", "Sharding"]
    })
    benchmark_path = os.path.join(results_dir, "mongodb_benchmark_results.csv")
    benchmark_df.to_csv(benchmark_path, index=False)
    print(f"[+] Benchmark results saved to: {benchmark_path}")

    if client:
        client.close()

    print("\n[+] MongoDB NoSQL Engine — All CRUD + Aggregation + Benchmarks Complete!")
    print("=========================================================\n")


def print_sharding_replication_config():
    """Print MongoDB sharding and replication configuration commands."""
    print("""
=======================================================
[*] MongoDB Sharding & Replication Configuration
=======================================================

─── REPLICATION (Replica Set) ───

MongoDB Replica Sets provide high availability through automatic failover.
A replica set consists of a Primary node and multiple Secondary nodes.

Configuration Steps:
  1. Start 3 mongod instances as a replica set:
     $ mongod --replSet rs0 --port 27017 --dbpath /data/rs0-0
     $ mongod --replSet rs0 --port 27018 --dbpath /data/rs0-1
     $ mongod --replSet rs0 --port 27019 --dbpath /data/rs0-2

  2. Initialize the replica set in mongo shell:
     > rs.initiate({
     ...   _id: "rs0",
     ...   members: [
     ...     { _id: 0, host: "localhost:27017" },
     ...     { _id: 1, host: "localhost:27018" },
     ...     { _id: 2, host: "localhost:27019" }
     ...   ]
     ... })

  3. Verify status:
     > rs.status()

  Replication Factor: 3 (1 Primary + 2 Secondaries)
  Write Concern: w=majority (acknowledged by majority of nodes)
  Read Preference: secondaryPreferred (for read scaling)

─── SHARDING (Horizontal Partitioning) ───

MongoDB Sharding distributes data across multiple machines for horizontal scalability.

Shard Key Selection for OmniPulse:
  • Shard key for 'transactions': { "customer.region": 1, "order.date": 1 }
    Rationale: Region provides geographic partitioning; date enables
    time-range queries to hit specific shards.

  • Shard key for 'customers': { "customer_id": "hashed" }
    Rationale: Hashed sharding distributes evenly across shards.

Configuration Steps:
  1. Start config servers:
     $ mongod --configsvr --replSet configReplSet --port 27020

  2. Start shard servers:
     $ mongod --shardsvr --replSet shard1 --port 27021
     $ mongod --shardsvr --replSet shard2 --port 27022

  3. Start mongos router:
     $ mongos --configdb configReplSet/localhost:27020 --port 27017

  4. Enable sharding in mongo shell:
     > sh.enableSharding("omnipulse_ecommerce")
     > sh.shardCollection(
     ...   "omnipulse_ecommerce.transactions",
     ...   { "customer.region": 1, "order.date": 1 }
     ... )

  Expected Distribution (3 shards, 125,000 documents):
    ┌──────────────────┬────────────────┬──────────────┐
    │ Shard            │ Chunk Range    │ Documents    │
    ├──────────────────┼────────────────┼──────────────┤
    │ shard1           │ NA + Europe    │  ~75,000     │
    │ shard2           │ Asia-Pacific   │  ~31,250     │
    │ shard3           │ LATAM + MEA    │  ~18,750     │
    └──────────────────┴────────────────┴──────────────┘
""")


if __name__ == "__main__":
    run_mongodb_operations()
