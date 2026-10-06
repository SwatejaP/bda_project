#!/usr/bin/env python3
"""
Hadoop MapReduce Job 1: Sales & Revenue by Product Category
Reducer: Aggregates total revenue and purchase counts per category.
"""

import sys

def reducer():
    current_category = None
    total_revenue = 0.0
    total_items = 0
    total_orders = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            category, values = line.split('\t', 1)
            amount_str, qty_str, count_str = values.split(',')
            
            amount = float(amount_str)
            qty = int(qty_str)
            count = int(count_str)
        except ValueError:
            continue

        if current_category == category:
            total_revenue += amount
            total_items += qty
            total_orders += count
        else:
            if current_category is not None:
                aov = total_revenue / total_orders if total_orders > 0 else 0
                print(f"{current_category}\t{total_revenue:.2f},{total_items},{total_orders},{aov:.2f}")
            current_category = category
            total_revenue = amount
            total_items = qty
            total_orders = count

    # Print the last category
    if current_category is not None:
        aov = total_revenue / total_orders if total_orders > 0 else 0
        print(f"{current_category}\t{total_revenue:.2f},{total_items},{total_orders},{aov:.2f}")

if __name__ == "__main__":
    reducer()
