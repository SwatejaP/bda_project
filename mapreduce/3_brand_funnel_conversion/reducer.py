#!/usr/bin/env python3
"""
Hadoop MapReduce Job 3: Brand Funnel Conversion (Kaggle Schema)
Reducer: Aggregates views, carts, and purchases per brand, calculating conversion rates.
"""

import sys

def reducer():
    current_brand = None
    views = 0
    carts = 0
    purchases = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            brand, values = line.split('\t', 1)
            event_type, count_str = values.split(',')
            count = int(count_str)
        except ValueError:
            continue

        if current_brand == brand:
            if event_type == "view": views += count
            elif event_type == "cart": carts += count
            elif event_type == "purchase": purchases += count
        else:
            if current_brand is not None:
                cart_conversion = (carts / views * 100) if views > 0 else 0.0
                purchase_conversion = (purchases / views * 100) if views > 0 else 0.0
                print(f"{current_brand}\tViews:{views}, Carts:{carts}, Purchases:{purchases}, CartRate:{cart_conversion:.2f}%, PurchaseRate:{purchase_conversion:.2f}%")
                
            current_brand = brand
            views = count if event_type == "view" else 0
            carts = count if event_type == "cart" else 0
            purchases = count if event_type == "purchase" else 0

    if current_brand is not None:
        cart_conversion = (carts / views * 100) if views > 0 else 0.0
        purchase_conversion = (purchases / views * 100) if views > 0 else 0.0
        print(f"{current_brand}\tViews:{views}, Carts:{carts}, Purchases:{purchases}, CartRate:{cart_conversion:.2f}%, PurchaseRate:{purchase_conversion:.2f}%")

if __name__ == "__main__":
    reducer()
