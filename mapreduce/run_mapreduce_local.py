"""
MapReduce Local Execution Engine & Pipeline Simulator
Simulates the Hadoop Streaming Pipeline:
Raw Data Stream -> Mapper -> Shuffle & Sort by Key -> Reducer -> Output Storage & Verification
"""

import os
import sys
import time
import subprocess
import pandas as pd

def run_job(job_name: str, mapper_path: str, reducer_path: str, input_csv: str, output_file: str):
    print(f"\n=======================================================")
    print(f"[*] Starting MapReduce Job: {job_name}")
    print(f"    Input:   {input_csv}")
    print(f"    Mapper:  {mapper_path}")
    print(f"    Reducer: {reducer_path}")
    print(f"=======================================================")

    t0 = time.time()
    python_exe = sys.executable

    # Step 1: Execute Mapper on raw input CSV
    t_map_start = time.time()
    with open(input_csv, "r", encoding="utf-8") as f_in:
        p_map = subprocess.Popen(
            [python_exe, mapper_path],
            stdin=f_in,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        map_out, map_err = p_map.communicate()
    
    if p_map.returncode != 0:
        print(f"[!] Error in Mapper: {map_err}")
        return False
    t_map_end = time.time()
    print(f"[+] Map Phase Complete in {t_map_end - t_map_start:.3f}s (Emitted {len(map_out.splitlines()):,} intermediate key-values)")

    # Step 2: Shuffle and Sort (Hadoop Shuffle Phase simulation)
    t_sort_start = time.time()
    lines = [line for line in map_out.splitlines() if line.strip()]
    lines.sort() # Standard Hadoop alphanumeric key sort
    sorted_input = "\n".join(lines) + "\n"
    t_sort_end = time.time()
    print(f"[+] Shuffle & Sort Phase Complete in {t_sort_end - t_sort_start:.3f}s")

    # Step 3: Execute Reducer on sorted stream
    t_red_start = time.time()
    p_red = subprocess.Popen(
        [python_exe, reducer_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    red_out, red_err = p_red.communicate(input=sorted_input)
    
    if p_red.returncode != 0:
        print(f"[!] Error in Reducer: {red_err}")
        return False
    t_red_end = time.time()
    print(f"[+] Reduce Phase Complete in {t_red_end - t_red_start:.3f}s")

    # Step 4: Write Output and Display
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(red_out)

    t_total = time.time() - t0
    print(f"[+] Job '{job_name}' FINISHED successfully in {t_total:.3f}s")
    print(f"[+] Results saved to: {output_file}\n")
    print("--- Output Snippet ---")
    print("\n".join(red_out.splitlines()[:15]))
    print("-----------------------\n")
    return True

def run_all_mapreduce_jobs():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, "data", "raw", "ecommerce_transactions.csv")
    
    if not os.path.exists(input_csv):
        print(f"[!] Raw dataset not found at {input_csv}. Generating now...")
        from data_generator import generate_ecommerce_dataset
        generate_ecommerce_dataset(125000, input_csv)

    mr_dir = os.path.join(base_dir, "mapreduce")
    proc_dir = os.path.join(base_dir, "data", "processed")

    jobs = [
        {
            "name": "Job 1: Category Sales & Revenue Aggregator",
            "mapper": os.path.join(mr_dir, "1_sales_by_category", "mapper.py"),
            "reducer": os.path.join(mr_dir, "1_sales_by_category", "reducer.py"),
            "output": os.path.join(proc_dir, "mapreduce_category_sales.tsv")
        },
        {
            "name": "Job 2: Customer RFM Aggregator",
            "mapper": os.path.join(mr_dir, "2_customer_rfm", "mapper.py"),
            "reducer": os.path.join(mr_dir, "2_customer_rfm", "reducer.py"),
            "output": os.path.join(proc_dir, "mapreduce_customer_rfm.tsv")
        },
        {
            "name": "Job 3: Brand Funnel Conversion",
            "mapper": os.path.join(mr_dir, "3_brand_funnel_conversion", "mapper.py"),
            "reducer": os.path.join(mr_dir, "3_brand_funnel_conversion", "reducer.py"),
            "output": os.path.join(proc_dir, "mapreduce_brand_funnel.tsv")
        }
    ]

    for job in jobs:
        run_job(job["name"], job["mapper"], job["reducer"], input_csv, job["output"])

if __name__ == "__main__":
    run_all_mapreduce_jobs()
