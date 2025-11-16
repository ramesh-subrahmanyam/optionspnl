"""
Earnings Database Management Module

This module contains functions to populate and maintain the earnings dates database
using the Finnhub API. The database is organized into monthly CSV files (M1.csv - M12.csv)
stored in the data/earnings_dates/ directory.

Usage:
    from manage_earnings_database import setup, store_symbols

    # Initialize the database (WARNING: this wipes existing data)
    setup()

    # Populate with symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    store_symbols(symbols, '2024-01-01', '2024-12-31')

Command-line usage:
    python manage_earnings_database.py --update   # Update for next 6 months
    python manage_earnings_database.py --reset    # Wipe and rebuild
    python manage_earnings_database.py --add TSLA # Add new symbol
"""

import time
import pandas as pd
import finnhub
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Finnhub API Key from environment
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY not found in .env file")

# Set up the Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Known ETFs and instruments that don't report earnings
KNOWN_NON_STOCKS = {
    # ETFs
    'ARKK', 'DIA', 'DXJ', 'EEM', 'EFV', 'EPOL', 'EWL', 'EWU', 'EWW', 'EWY',
    'EWZ', 'FBT', 'FLIN', 'FXI', 'GBTC', 'GDX', 'GLD', 'IBB', 'IGIB', 'IJH',
    'IJR', 'IJS', 'IUSB', 'IVOG', 'ITA', 'ITB', 'IWM', 'IYR', 'KOLD', 'MDY',
    'QLTA', 'QQQ', 'QUAL', 'SMH', 'SMIN', 'SPXL', 'SPY', 'TAN', 'TLT', 'TMF',
    'UGA', 'URA', 'USO', 'VBR', 'VCIT', 'VCLT', 'VFMV', 'VGK', 'VIOG', 'VNQ',
    'VO', 'VRT', 'VT', 'VTI', 'VTV', 'VUG', 'XBI', 'XHB', 'XLC', 'XLE', 'XLI',
    'XLK', 'XLP', 'XLV', 'XLY', 'XME', 'XOP', 'XRT',
    # Bond/Cash ETFs
    'BNDX', 'BOXX', 'DBND', 'BHYAX',
    # Crypto/Commodities
    'ETH',
}


def is_valid_symbol(symbol):
    """
    Check if a symbol is likely valid for earnings data.

    Args:
        symbol (str): Stock symbol to validate

    Returns:
        bool: True if symbol appears valid, False otherwise
    """
    if not symbol or not symbol.strip():
        return False

    # Check against known non-stocks list
    if symbol.upper() in KNOWN_NON_STOCKS:
        return False

    # Check for spaces (invalid for stock symbols)
    if ' ' in symbol:
        return False

    # Check for common ETF/index suffixes that typically don't have earnings
    etf_keywords = ['ETF', 'FUND', 'INDEX', 'TRUST']
    if any(keyword in symbol.upper() for keyword in etf_keywords):
        return False

    # Most valid stock symbols are 1-5 characters
    # Longer ones are often ETFs or special instruments
    if len(symbol) > 5:
        return False

    # Check for common non-stock patterns
    invalid_patterns = ['ACCOUNT', 'TOTAL', 'CASH', 'BALANCE']
    if any(pattern in symbol.upper() for pattern in invalid_patterns):
        return False

    return True


def get_earnings_dates(symbol, start_date, end_date):
    """
    Fetch earnings dates for a symbol from Finnhub API.

    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame containing earnings calendar data
    """
    earnings = finnhub_client.earnings_calendar(
        _from=start_date,
        to=end_date,
        symbol=symbol
    )
    return pd.DataFrame(earnings['earningsCalendar'])


def setup():
    """
    Initialize the earnings dates database.

    WARNING: This wipes out the existing database and creates fresh monthly CSV files.
    Creates files data/earnings_dates/M1.csv through data/earnings_dates/M12.csv
    """
    # Create directory if it doesn't exist
    os.makedirs('data/earnings_dates', exist_ok=True)

    for i in range(1, 13):
        filename = f"data/earnings_dates/M{i}.csv"
        with open(filename, "w") as h:
            h.write("symbol, date\n")
        print(f"Initialized {filename}")


def store(symbol, dt):
    """
    Store a single earnings date for a symbol in the appropriate monthly file.

    Args:
        symbol (str): Stock symbol
        dt (str): Earnings date in 'YYYY-MM-DD' format
    """
    # Extract month from date (assumes format YYYY-MM-DD)
    m = int(dt[-5:-3])
    fname = f"data/earnings_dates/M{m}.csv"
    print(f"{fname}: {symbol}, {dt}")

    with open(fname, "a") as h:
        h.write(f"{symbol},  {dt}\n")


def store_symbols(symbols, start_date, end_date, delay_seconds=2):
    """
    Fetch and store earnings dates for multiple symbols.

    Args:
        symbols (list): List of stock symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        delay_seconds (float): Delay between API calls to avoid rate limiting (default: 2)

    Returns:
        dict: Dictionary with symbols as keys and status (success/failed/skipped) as values
    """
    results = {}

    for i, symbol in enumerate(symbols):
        # Validate symbol before making API call
        if not is_valid_symbol(symbol):
            print(f"⊗ {symbol}: skipped (likely ETF/index/invalid)")
            results[symbol] = 'skipped'
            continue

        try:
            earnings_df = get_earnings_dates(symbol, start_date, end_date)

            # Check if DataFrame is empty or has no 'date' column
            if earnings_df.empty or 'date' not in earnings_df.columns:
                print(f"⚠ {symbol}: no earnings data available")
                results[symbol] = 'no_data'
            else:
                dts = earnings_df['date']

                for dt in dts:
                    store(symbol, dt)

                results[symbol] = 'success'
                print(f"✓ {symbol}: stored {len(dts)} earnings dates")

        except Exception as e:
            error_msg = str(e)
            # Shorten long error messages
            if len(error_msg) > 60:
                error_msg = error_msg[:60] + "..."
            print(f"✗ {symbol}: {error_msg}")
            results[symbol] = 'failed'

        # Add delay between requests (except after the last one)
        # Only delay if we actually made an API call
        if results.get(symbol) != 'skipped' and i < len(symbols) - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    return results


def update_symbols(symbols, start_date, end_date, delay_seconds=2):
    """
    Update earnings dates for symbols without wiping the database.
    This is an alias for store_symbols for clarity.

    Args:
        symbols (list): List of stock symbols to update
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        delay_seconds (float): Delay between API calls (default: 2)

    Returns:
        dict: Dictionary with symbols as keys and status as values
    """
    return store_symbols(symbols, start_date, end_date, delay_seconds)


def get_symbols_needing_update(symbols, start_date, end_date):
    """
    Identify symbols that don't have earnings data for the specified date range.

    Args:
        symbols (list): List of symbols to check
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        list: Symbols that need updating
    """
    from datetime import datetime

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Load all earnings data
    all_earnings = {}
    for i in range(1, 13):
        filename = f'data/earnings_dates/M{i}.csv'
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename, skipinitialspace=True)
                if len(df) > 0:
                    df.columns = df.columns.str.strip()
                    df['date'] = pd.to_datetime(df['date'])
                    for _, row in df.iterrows():
                        symbol = row['symbol']
                        date = row['date']
                        if symbol not in all_earnings:
                            all_earnings[symbol] = []
                        all_earnings[symbol].append(date)
            except Exception as e:
                print(f"Warning: Error reading {filename}: {e}")

    # Check which symbols need updates
    symbols_to_update = []
    for symbol in symbols:
        if symbol not in all_earnings:
            # No data at all for this symbol
            symbols_to_update.append(symbol)
        else:
            # Check if there's data in the date range
            symbol_dates = all_earnings[symbol]
            dates_in_range = [d for d in symbol_dates if start <= d <= end]
            if len(dates_in_range) == 0:
                # No earnings data in the specified range
                symbols_to_update.append(symbol)

    return symbols_to_update


def add_symbol_to_stocks_list(symbol, stocks_list_file='data/stocks_list.csv'):
    """
    Add a symbol to the stocks_list file if it's not already there.

    Args:
        symbol (str): Stock symbol to add
        stocks_list_file (str): Path to stocks_list file

    Returns:
        bool: True if added, False if already exists
    """
    # Read existing symbols
    existing_symbols = []
    if os.path.exists(stocks_list_file):
        with open(stocks_list_file, 'r') as f:
            existing_symbols = [line.strip() for line in f.readlines()]

    # Check if symbol already exists
    if symbol in existing_symbols:
        return False

    # Add symbol
    existing_symbols.append(symbol)
    existing_symbols.sort()

    # Write back
    with open(stocks_list_file, 'w') as f:
        for sym in existing_symbols:
            f.write(sym + '\n')

    return True


def update_earnings_database(symbols=None, months_forward=6):
    """
    Update earnings database for specified symbols or from stocks_list.csv.

    This is a programmatic API for updating the earnings database, designed to be
    called from other Python code (e.g., from screener.py).

    Args:
        symbols (list, optional): List of symbols to update. If None, reads from data/stocks_list.csv
        months_forward (int): Number of months forward to fetch earnings data (default: 6)

    Returns:
        dict: Dictionary with symbols as keys and status (success/no_data/failed/skipped) as values
    """
    from datetime import datetime, timedelta

    # Load symbols from file if not provided
    if symbols is None:
        stocks_list_file = 'data/stocks_list.csv'
        if not os.path.exists(stocks_list_file):
            raise FileNotFoundError(f"{stocks_list_file} not found!")

        with open(stocks_list_file, 'r') as f:
            symbols = [line.strip() for line in f.readlines() if line.strip()]

    # Calculate date range
    today = datetime.now()
    future_date = today + timedelta(days=months_forward * 30)
    start_date = today.strftime('%Y-%m-%d')
    end_date = future_date.strftime('%Y-%m-%d')

    # Find symbols that need updating
    symbols_to_update = get_symbols_needing_update(symbols, start_date, end_date)

    if len(symbols_to_update) == 0:
        print("✓ All symbols are up to date!")
        return {}

    print(f"Updating {len(symbols_to_update)} symbols (skipping {len(symbols) - len(symbols_to_update)} current symbols)")

    # Update symbols
    results = store_symbols(symbols_to_update, start_date, end_date)

    return results


# Command-line interface
if __name__ == "__main__":
    import sys
    import argparse
    from datetime import datetime, timedelta

    parser = argparse.ArgumentParser(
        description='Manage earnings database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update earnings for next 6 months (only symbols that need it)
  python manage_earnings_database.py --update

  # Reset and rebuild entire database
  python manage_earnings_database.py --reset

  # Add a new symbol to stocks_list and update its earnings
  python manage_earnings_database.py --add TSLA
        """
    )

    parser.add_argument('--update', action='store_true',
                        help='Update earnings database for next 6 months (only symbols needing update)')
    parser.add_argument('--reset', action='store_true',
                        help='Wipe and rebuild entire earnings database')
    parser.add_argument('--add', metavar='SYMBOL',
                        help='Add symbol to stocks_list and update its earnings')

    args = parser.parse_args()

    # Check if no arguments provided
    if not (args.update or args.reset or args.add):
        parser.print_help()
        sys.exit(1)

    print("Earnings Database Manager")
    print("=" * 60)

    # Load stocks list
    stocks_list_file = 'data/stocks_list.csv'

    # --update: Update earnings for next 6 months (selective)
    if args.update:
        print("\n📅 UPDATE MODE: Updating earnings for next 6 months")
        print("-" * 60)

        # Load symbols from stocks_list
        if not os.path.exists(stocks_list_file):
            print(f"Error: {stocks_list_file} not found!")
            sys.exit(1)

        with open(stocks_list_file, 'r') as f:
            symbols = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Loaded {len(symbols)} symbols from {stocks_list_file}")

        # Calculate date range (today to 6 months from now)
        today = datetime.now()
        six_months = today + timedelta(days=180)
        start_date = today.strftime('%Y-%m-%d')
        end_date = six_months.strftime('%Y-%m-%d')

        print(f"Date range: {start_date} to {end_date}")

        # Find symbols that need updating
        print("\nChecking which symbols need updates...")
        symbols_to_update = get_symbols_needing_update(symbols, start_date, end_date)

        if len(symbols_to_update) == 0:
            print("✓ All symbols are up to date!")
            sys.exit(0)

        print(f"Found {len(symbols_to_update)} symbols needing updates")
        print(f"Skipping {len(symbols) - len(symbols_to_update)} symbols (already current)")

        # Update
        print(f"\nUpdating {len(symbols_to_update)} symbols...")
        results = store_symbols(symbols_to_update, start_date, end_date)

        # Summary
        success_count = sum(1 for status in results.values() if status == 'success')
        no_data_count = sum(1 for status in results.values() if status == 'no_data')
        failed_count = sum(1 for status in results.values() if status == 'failed')
        skipped_count = sum(1 for status in results.values() if status == 'skipped')
        print("\n" + "=" * 60)
        print(f"Update complete:")
        print(f"  ✓ Success: {success_count}")
        print(f"  ⚠ No data: {no_data_count}")
        print(f"  ⊗ Skipped: {skipped_count}")
        print(f"  ✗ Failed: {failed_count}")
        print(f"  Total: {len(symbols_to_update)}")

    # --reset: Wipe and rebuild database
    elif args.reset:
        print("\n⚠️  RESET MODE: Wiping and rebuilding database")
        print("-" * 60)

        # Confirm
        response = input("This will delete all existing earnings data. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            sys.exit(0)

        # Load symbols
        if not os.path.exists(stocks_list_file):
            print(f"Error: {stocks_list_file} not found!")
            sys.exit(1)

        with open(stocks_list_file, 'r') as f:
            symbols = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Loaded {len(symbols)} symbols from {stocks_list_file}")

        # Calculate date range (today to 6 months from now)
        today = datetime.now()
        six_months = today + timedelta(days=180)
        start_date = today.strftime('%Y-%m-%d')
        end_date = six_months.strftime('%Y-%m-%d')

        # Setup (wipe database)
        print("\nInitializing database...")
        setup()

        # Update all symbols
        print(f"\nUpdating all {len(symbols)} symbols...")
        print(f"Date range: {start_date} to {end_date}")
        results = store_symbols(symbols, start_date, end_date)

        # Summary
        success_count = sum(1 for status in results.values() if status == 'success')
        no_data_count = sum(1 for status in results.values() if status == 'no_data')
        failed_count = sum(1 for status in results.values() if status == 'failed')
        skipped_count = sum(1 for status in results.values() if status == 'skipped')
        print("\n" + "=" * 60)
        print(f"Reset complete:")
        print(f"  ✓ Success: {success_count}")
        print(f"  ⚠ No data: {no_data_count}")
        print(f"  ⊗ Skipped: {skipped_count}")
        print(f"  ✗ Failed: {failed_count}")
        print(f"  Total: {len(symbols)}")

    # --add SYMBOL: Add new symbol
    elif args.add:
        symbol = args.add.upper()
        print(f"\n➕ ADD MODE: Adding symbol {symbol}")
        print("-" * 60)

        # Add to stocks_list
        added = add_symbol_to_stocks_list(symbol, stocks_list_file)
        if added:
            print(f"✓ Added {symbol} to {stocks_list_file}")
        else:
            print(f"ℹ {symbol} already in {stocks_list_file}")

        # Update earnings for this symbol
        today = datetime.now()
        six_months = today + timedelta(days=180)
        start_date = today.strftime('%Y-%m-%d')
        end_date = six_months.strftime('%Y-%m-%d')

        print(f"\nUpdating earnings for {symbol}...")
        print(f"Date range: {start_date} to {end_date}")

        # Ensure database is initialized
        if not os.path.exists('data/earnings_dates/M1.csv'):
            print("Initializing database...")
            setup()

        results = store_symbols([symbol], start_date, end_date)

        # Summary
        status = results.get(symbol, 'failed')
        status_icon = "✓" if status == "success" else "✗"
        print(f"\n{status_icon} {symbol}: {status}")
