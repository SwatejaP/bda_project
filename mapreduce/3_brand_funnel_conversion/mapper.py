#!/usr/bin/env python3
"""
Hadoop MapReduce Job 3: Brand Funnel Conversion (Kaggle Schema)
Mapper: Emits (brand \t event_type, 1)
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
            brand = row[5].strip()
            
            if not brand:
                brand = "unknown_brand"
                
            print(f"{brand}\t{event_type},1")

        except (IndexError, ValueError):
            continue

if __name__ == "__main__":
    mapper()
