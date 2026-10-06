#!/usr/bin/env python3
"""
Hadoop MapReduce Job 2: Customer RFM Engine
Reducer: Aggregates frequency, monetary value, and recency per user.
"""

import sys
from datetime import datetime

def reducer():
    current_user = None
    total_spend = 0.0
    purchase_count = 0
    last_purchase_date = None
    
    # Assuming analysis baseline date is May 1st 2020 since Kaggle data ends April 2020
    # For our synthetic data, we can just use the current time or a fixed date.
    # Let's dynamically find the max date in the data, or use a fixed baseline.
    baseline_date = datetime(2020, 5, 1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            user_id, values = line.split('\t', 1)
            date_str, price_str = values.split(',')
            
            price = float(price_str)
            # handle formats like YYYY-MM-DD HH:MM:SS
            purchase_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            
        except ValueError:
            continue

        if current_user == user_id:
            total_spend += price
            purchase_count += 1
            if last_purchase_date is None or purchase_date > last_purchase_date:
                last_purchase_date = purchase_date
        else:
            if current_user is not None:
                # Calculate Recency in days
                recency = (baseline_date - last_purchase_date).days if last_purchase_date else 999
                # Output: user_id \t Recency, Frequency, Monetary
                print(f"{current_user}\t{recency},{purchase_count},{total_spend:.2f}")
                
            current_user = user_id
            total_spend = price
            purchase_count = 1
            last_purchase_date = purchase_date

    if current_user is not None:
        recency = (baseline_date - last_purchase_date).days if last_purchase_date else 999
        print(f"{current_user}\t{recency},{purchase_count},{total_spend:.2f}")

if __name__ == "__main__":
    reducer()
