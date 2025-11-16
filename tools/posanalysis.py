import sys
import os
import pandas as pd

# Add project root to Python path to enable imports
# This works whether running from project root or tools directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from libs.positionslib import get_expiration_indexed_positions

def main():
    NEEDED_COLUMNS=['Symbol', 'Quantity', 'Cost Basis', "PnL", "Margin", "Margin Requirement"]

    filename="data/pos.csv"

    dct=get_expiration_indexed_positions(filename, needed_columns=NEEDED_COLUMNS)

    # Convert dictionary to DataFrame
    rows = []
    for date, df in dct.items():
        row = {
            'Date': date,
            'Cost Basis': int(-df['Cost Basis'].sum()),
            'Quantity': int(-df['Quantity'].sum()),
            'Premium': int(-df['Cost Basis'].sum().round()),  # Premium = Cost Basis
            'PnL': int(df['PnL'].sum()),
            'Margin': int(df['Margin'].sum())
        }
        rows.append(row)

    # Create DataFrame indexed by date
    weekly_df = pd.DataFrame(rows)
    weekly_df['Date'] = pd.to_datetime(weekly_df['Date'])
    weekly_df = weekly_df.set_index('Date')
    weekly_df = weekly_df.sort_index()

    print("Weekly DataFrame:")
    print(weekly_df)
    print("\n")

    # Create monthly aggregated DataFrame
    monthly_df = weekly_df.copy()
    monthly_df['Month'] = monthly_df.index.to_period('M')
    monthly_agg = monthly_df.groupby('Month').agg({
        'Cost Basis': 'sum',
        'Quantity': 'sum',
        'Premium': 'sum',
        'PnL': 'sum',
        'Margin': 'sum'
    })

    print("Monthly Aggregated DataFrame:")
    print(monthly_agg)
    return dct

if __name__ == "__main__":
    main()
    