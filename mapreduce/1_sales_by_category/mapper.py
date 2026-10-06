#!/usr/bin/env python3
"""
Hadoop MapReduce Job 1: Sales & Revenue by Product Category (Kaggle Schema)
Mapper: Reads raw CSV lines from sys.stdin, parses fields, filters for 'purchase' events, 
and emits (Category_Code, Price, 1, 1).
"""

import sys
import csv

def mapper():
    reader = csv.reader(sys.stdin)
    header_skipped = False

    for row in reader:
        if not row:
            continue
        
        # Skip header line
        if not header_skipped and ("event_time" in row[0] or "event_type" in row):
            header_skipped = True
            continue

        try:
            # Expected schema: event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session
            event_type = row[1].strip()
            if event_type != "purchase":
                continue  # Only aggregate actual purchases
            
            category = row[4].strip()
            if not category:
                category = "unknown_category"
                
            price = float(row[6].strip())

            # Emit tab-separated: Category \t Price,1,1 (Simulating Amount,Quantity,Count to match old reducer style)
            print(f"{category}\t{price:.2f},1,1")

        except (IndexError, ValueError):
            continue

if __name__ == "__main__":
    mapper()
