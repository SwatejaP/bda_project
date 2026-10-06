#!/usr/bin/env python3
"""
Hadoop MapReduce Job 2: Customer RFM (Recency, Frequency, Monetary) Engine (Kaggle Schema)
Mapper: Filters for 'purchase' events and emits (user_id \t event_time,price)
"""

import sys
import csv

def mapper():
    reader = csv.reader(sys.stdin)
    header_skipped = False

    for row in reader:
        if not row:
            continue
        
        if not header_skipped and ("event_time" in row[0] or "event_type" in row):
            header_skipped = True
            continue

        try:
            # Schema: event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session
            event_type = row[1].strip()
            if event_type != "purchase":
                continue
            
            event_time = row[0].strip()
            # Extract just the date/time string, dropping ' UTC' if present for easier parsing
            if "UTC" in event_time:
                event_time = event_time.replace(" UTC", "").strip()
                
            price = float(row[6].strip())
            user_id = row[7].strip()

            print(f"{user_id}\t{event_time},{price:.2f}")

        except (IndexError, ValueError):
            continue

if __name__ == "__main__":
    mapper()
