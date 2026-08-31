import pandas as pd
import os

# Check all CSV files in directory
files = ['predictions_vs_actuals_full.csv', 'predictions_vs_actuals.csv', 'predictions.csv', 'test_df.csv']
for f in files:
    if os.path.exists(f):
        print(f"\n{'='*60}")
        print(f"File: {f}")
        print(f"{'='*60}")
        df = pd.read_csv(f)
        print(f"Columns: {list(df.columns)[:10]}")
        print(f"Shape: {df.shape}")
        print("\nFirst row:")
        print(df.iloc[0] if len(df) > 0 else "Empty")
        if f == 'test_df.csv':
            print(f"\nTarget column (CO(GT)) sample:")
            print(df['CO(GT)'].head(10).values)
