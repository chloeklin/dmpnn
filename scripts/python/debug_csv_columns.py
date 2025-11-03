#!/usr/bin/env python3
"""
Debug script to check columns in consolidated CSV files
"""

import pandas as pd
from pathlib import Path

def check_csv_columns(csv_path: Path):
    """Check what columns are in a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"\n📊 {csv_path.name}")
        print(f"   Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"   Sample row: {df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")

combined_dir = Path('plots/combined')
csv_files = list(combined_dir.glob('*_consolidated_results.csv'))

print(f"🔍 Found {len(csv_files)} consolidated results files")
for csv_file in sorted(csv_files):
    check_csv_columns(csv_file)
