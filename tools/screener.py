
import sys, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io
from contextlib import redirect_stdout
import logging
import yfinance as yf
import os

# Add project root to Python path to enable imports
# This works whether running from project root or tools directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from libs.historical_cache import HistoricalCache

# Helper function to get data directory path
def _get_data_path(*parts):
    """Get absolute path to data directory or subdirectory."""
    return os.path.join(project_root, 'data', *parts)



# Earnings database management functions have been moved to manage_earnings_database.py
# Import them if needed: from manage_earnings_database import get_earnings_dates, setup, store, store_symbols


def list_stocks():
    stocks = []

    # Loop through the files labeled M1 to M12
    for i in range(1, 13):
        file_name = _get_data_path('earnings_dates', f'M{i}.csv')
        try:
            df = pd.read_csv(file_name, skipinitialspace=True)
            if len(df) == 0: continue
            stocks.extend(df['symbol'].tolist())
        except FileNotFoundError:
            print(f"File {file_name} not found")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    return set(stocks)

 
def get_stocks_within_days(N):
    today = datetime.today()
    end_date = today + timedelta(days=N)
    stocks = []

    # Loop through the files labeled M1 to M12
    for i in range(1, 13):
        file_name = _get_data_path('earnings_dates', f'M{i}.csv')
        try:
            df = pd.read_csv(file_name, skipinitialspace=True)
            if len(df) == 0: continue
            df['date'] = pd.to_datetime(df['date'])
            filtered_df = df[(df['date'] >= today) & (df['date'] <= end_date)]
            stocks.extend(filtered_df['symbol'].tolist())
        except FileNotFoundError:
            print(f"File {file_name} not found")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    return stocks



def get_stocks_outside_days(N):
    today = datetime.today()
    end_date = today + timedelta(days=N)
    all_stocks = {}
    stocks_within_range = set()

    # Loop through the files labeled M1 to M12
    for i in range(1, 13):
        file_name = _get_data_path('earnings_dates', f'M{i}.csv')
        try:
            df = pd.read_csv(file_name, skipinitialspace=True)
            if len(df) == 0: continue
            
            # Strip spaces from column names
            df.columns = df.columns.str.strip()

            df['date'] = pd.to_datetime(df['date'])
            
            # Add all stocks and their dates to the all_stocks dictionary
            for index, row in df.iterrows():
                stock = row['symbol']
                date = row['date']
                if stock in all_stocks:
                    all_stocks[stock].append(date)
                else:
                    all_stocks[stock] = [date]
            
            # Identify stocks with earnings dates within the specified range
            stocks_within_range.update(df[(df['date'] >= today) & (df['date'] <= end_date)]['symbol'].tolist())
            
        except FileNotFoundError:
            print(f"File {file_name} not found")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    # Exclude stocks with earnings dates within the specified range
    stocks_outside_range = {stock: dates for stock, dates in all_stocks.items() if stock not in stocks_within_range}

    # Get the next earnings date for each stock and extract just the date
    result = [(stock, min(dates).date()) for stock, dates in stocks_outside_range.items()]

    return result

def get_stocks_outside_date(end_date_str):
    today = datetime.combine(datetime.today().date(), datetime.min.time())
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    all_stocks = {}
    stocks_within_range = set()

    # Loop through the files labeled M1 to M12
    for i in range(1, 13):
        file_name = _get_data_path('earnings_dates', f'M{i}.csv')
        try:
            df = pd.read_csv(file_name, skipinitialspace=True)
            if len(df) == 0: continue
            
            # Strip spaces from column names
            df.columns = df.columns.str.strip()

            df['date'] = pd.to_datetime(df['date'])
            
            # Add all stocks and their dates to the all_stocks dictionary
            for index, row in df.iterrows():
                stock = row['symbol']
                date = row['date']
                if stock in all_stocks:
                    all_stocks[stock].append(date)
                else:
                    all_stocks[stock] = [date]
            
            # Identify stocks with any earnings dates within the specified range
            in_range_stocks = df[(df['date'] >= today) & (df['date'] <= end_date)]['symbol'].unique()
            #print(i, "GM" in in_range_stocks, today, df[df.symbol=="GM"])
            stocks_within_range.update(in_range_stocks)
            
        except FileNotFoundError:
            print(f"File {file_name} not found")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    # Filter out stocks with any earnings dates within the specified range
    stocks_outside_range = {stock: dates for stock, dates in all_stocks.items() if stock not in stocks_within_range}

    # Get the next earnings date for each stock, strictly after end_date
    result = []
    for stock, dates in stocks_outside_range.items():
        future_dates = [date for date in dates if date > end_date]
        if future_dates:
            result.append((stock, min(future_dates).date()))

    return pd.DataFrame(result, columns=["symbol", "earnings_date"])

class StockData:
    def __init__(self, symbols, use_cache=True, staleness_days=1):
        """
        Initialize StockData with symbols and optional caching.

        :param symbols: List of stock symbols
        :param use_cache: Whether to use historical price cache (default: True)
        :param staleness_days: Cache staleness in days (default: 1)
        """
        self.symbols = symbols
        self.df = pd.DataFrame(index=symbols)
        self.use_cache = use_cache
        if use_cache:
            self.cache = HistoricalCache(staleness_days=staleness_days)

    def fetch_price_data(self, start_date, end_date):
        """
        Fetch historical price data from cache or Yahoo Finance.

        :param start_date: Start date for fetching data in 'YYYY-MM-DD' format.
        :param end_date: End date for fetching data in 'YYYY-MM-DD' format.
        :return: DataFrame with columns 'symbol', 'date', 'price'.
        """
        data = []

        if self.use_cache:
            # Use cache for data retrieval
            for symbol in self.symbols:
                stock_data = self.cache.lookup(symbol, auto_update=True)
                if stock_data is not None and len(stock_data) > 0:
                    # The cache returns a DataFrame with Date as index
                    # Reset index to make Date a column
                    stock_data = stock_data.copy()
                    stock_data.reset_index(inplace=True)

                    # Ensure Date column is datetime type
                    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

                    # Filter to date range
                    stock_data = stock_data[(stock_data['Date'] >= start_date) &
                                           (stock_data['Date'] <= end_date)]

                    if len(stock_data) > 0:
                        stock_data['symbol'] = symbol
                        data.append(stock_data[['symbol', 'Date', 'Close']])
        else:
            # Direct yfinance download without cache
            for symbol in self.symbols:
                stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(stock_data) > 0:
                    stock_data['symbol'] = symbol
                    stock_data.reset_index(inplace=True)
                    data.append(stock_data[['symbol', 'Date', 'Close']])

        if len(data) == 0:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['symbol', 'date', 'price'])

        combined_data = pd.concat(data)
        combined_data.columns = ['symbol', 'date', 'price']
        return combined_data
    
    def add_returns(self, N):
        """
        Add N-day returns to the DataFrame as a new column.
        
        :param N: Number of days for calculating returns.
        """
        # Fetch historical price data with a slightly extended range
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - pd.DateOffset(days=N*2)).strftime('%Y-%m-%d')
        price_data = self.fetch_price_data(start_date, end_date)
        
        # Pivot price_data to have symbols as index and dates as columns
        price_pivot = price_data.pivot(index='date', columns='symbol', values='price')
        
        # Calculate N-day returns
        n_day_returns = price_pivot.pct_change(periods=N, fill_method=None).fillna(0) * 100  # Convert to percentage
        
        # Convert N-day returns to integers (0 decimal places)
        n_day_returns = n_day_returns.round(1).astype(float)
        
         
        # Attach N-day returns to the main DataFrame
        self.df[f'ret{N}'] = n_day_returns.iloc[-1] 
        
        
    def attach_earnings_dates(self, earnings_dates):
        """
        Attach earnings dates to the DataFrame.

        :param earnings_dates: A DataFrame with columns 'symbol' and 'earnings_date'.
        """
        self.df["earnings_date"]=pd.to_datetime(earnings_dates.set_index("symbol").earnings_date).dt.date


def main(days_forward=30, update_earnings=True, output_file=None,
         stocks_list_file=None, use_cache=True, staleness_days=1):
    """
    Main screening workflow function.

    Args:
        days_forward (int): Number of days forward to check for earnings (default: 30)
        update_earnings (bool): Whether to update earnings database before screening (default: True)
        output_file (str): Path to save screening results CSV (default: 'data/screener_output.csv')
        stocks_list_file (str): Path to stocks list file (default: 'data/stocks_list.csv')
        use_cache (bool): Whether to use historical price cache (default: True)
        staleness_days (int): Cache staleness in days (default: 1)

    Returns:
        pd.DataFrame: Screening results sorted by 22-day returns
    """
    # Set default paths relative to project root
    if output_file is None:
        output_file = _get_data_path('screener_output.csv')
    if stocks_list_file is None:
        stocks_list_file = _get_data_path('stocks_list.csv')

    # Step 1: Load symbols from stocks_list.csv
    with open(stocks_list_file, 'r') as f:
        symbols = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Loaded {len(symbols)} symbols from {stocks_list_file}")

    # Step 2: Update earnings database (optional)
    if update_earnings:
        from tools.manage_earnings_database import update_earnings_database
        print("\nUpdating earnings database...")
        update_earnings_database(symbols)

    # Step 3: Screen for stocks with no earnings in next N days
    today = datetime.now()
    end_date_str = (today + timedelta(days=days_forward)).strftime('%Y-%m-%d')
    print(f"\nScreening for stocks with no earnings between now and {end_date_str}")

    earnings_df = get_stocks_outside_date(end_date_str)
    print(f"Found {len(earnings_df)} stocks with no earnings in the next {days_forward} days")

    # Step 4: Calculate returns for screened stocks (using historical cache)
    if len(earnings_df) > 0:
        # Create StockData with cache enabled
        stock_data = StockData(list(earnings_df.symbol), use_cache=use_cache, staleness_days=staleness_days)

        # Calculate returns over different periods
        print("\nCalculating returns (using cached historical data)...")
        stock_data.add_returns(5)   # 5-day (1 week) returns
        stock_data.add_returns(22)  # 22-day (1 month) returns
        stock_data.add_returns(66)  # 66-day (3 month) returns

        # Attach earnings dates
        stock_data.attach_earnings_dates(earnings_df)

        # Sort by 22-day returns
        results_df = stock_data.df.sort_values('ret22', ascending=False)

        # Save to CSV
        results_df.to_csv(output_file)
        print(f"\n✓ Results saved to {output_file}")

        # Display results
        print("\nScreening Results:")
        print("="*80)
        print(results_df)

        return results_df
    else:
        print("No stocks found matching criteria")
        return pd.DataFrame()

if __name__ == "__main__":
    main()