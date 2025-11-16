import os
import json
import time
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import logging

# Constants
TRADING_DAYS_PER_YEAR = 252

class HistoricalCache:
    """
    A cache for storing historical stock price data from Yahoo Finance.
    Each stock's data is stored in a separate JSON file for efficient updates.

    Features:
    - Configurable staleness threshold (default: 1 day)
    - Configurable history length (default: 252 days)
    - Configurable columns to store (default: Close only)
    - Lazy updates: only refreshes when data is requested and stale
    - Per-stock caching: different stocks can have different freshness
    """

    def __init__(self,
                 cache_dir='data/historical_cache',
                 staleness_days=1,
                 history_days=252,
                 columns=None,
                 data_source='yfinance'):
        """
        Initialize the historical cache.

        Args:
            cache_dir (str): Directory to store individual stock cache files
            staleness_days (int): Days before cache is considered stale (-1 means never update)
            history_days (int): Number of days of historical data to store
            columns (list): List of columns to store (e.g., ['Close', 'Open', 'Volume'])
            data_source (str): Data source to use (currently only 'yfinance' supported)
        """
        self.cache_dir = cache_dir
        self.staleness_days = staleness_days
        self.history_days = history_days
        self.columns = columns if columns is not None else ['Close']
        self.data_source = data_source

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _get_cache_file_path(self, symbol):
        """Get the file path for a symbol's cache file."""
        return os.path.join(self.cache_dir, f"{symbol}.json")

    def _load_cache_file(self, symbol):
        """
        Load cached data for a symbol from disk.

        Args:
            symbol (str): Stock symbol

        Returns:
            dict: Cached data or None if file doesn't exist or is corrupted
        """
        cache_file = self._get_cache_file_path(symbol)

        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Corrupted cache file for {symbol}: {e}")
            # Delete corrupted file
            try:
                os.remove(cache_file)
            except:
                pass
            return None

    def _save_cache_file(self, symbol, data):
        """
        Save cached data for a symbol to disk.

        Args:
            symbol (str): Stock symbol
            data (dict): Data to cache
        """
        cache_file = self._get_cache_file_path(symbol)

        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Cache updated for {symbol} at {datetime.now()}")
        except IOError as e:
            self.logger.error(f"Error saving cache for {symbol}: {e}")

    def needs_update(self, symbol):
        """
        Check if a symbol's cache needs to be updated.

        Args:
            symbol (str): Stock symbol

        Returns:
            bool: True if cache needs update, False otherwise
        """
        cached_data = self._load_cache_file(symbol)

        # No cache file exists
        if cached_data is None:
            return True

        # If staleness_days is -1, never update (cache exists)
        if self.staleness_days == -1:
            return False

        # Check if columns have changed
        cached_columns = set(cached_data.get('columns', []))
        requested_columns = set(self.columns)
        if cached_columns != requested_columns:
            return True

        # Check if last_updated exists
        if 'last_updated' not in cached_data:
            return True

        # Check staleness
        try:
            last_updated = datetime.fromisoformat(cached_data['last_updated'])
            current_time = datetime.now()
            time_diff = current_time - last_updated

            # Calculate days as a float for more precise comparison
            days_old = time_diff.total_seconds() / (24 * 3600)

            return days_old > self.staleness_days
        except (ValueError, KeyError):
            return True

    def _fetch_from_yfinance(self, symbol):
        """
        Fetch historical data from Yahoo Finance.

        Args:
            symbol (str): Stock symbol

        Returns:
            pd.DataFrame: Historical data with date index
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            # Add extra days to account for weekends/holidays
            start_date = end_date - timedelta(days=int(self.history_days * 1.5))

            # Fetch data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return None

            # Keep only the requested columns
            available_columns = [col for col in self.columns if col in hist.columns]
            if not available_columns:
                self.logger.warning(f"None of the requested columns found for {symbol}")
                return None

            hist = hist[available_columns]

            # Keep only the most recent history_days trading days
            hist = hist.tail(self.history_days)

            return hist

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def _fetch_data(self, symbol):
        """
        Fetch data from the configured data source.

        Args:
            symbol (str): Stock symbol

        Returns:
            pd.DataFrame: Historical data
        """
        if self.data_source == 'yfinance':
            return self._fetch_from_yfinance(symbol)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

    def update(self, symbol):
        """
        Update cache for a single symbol.

        Args:
            symbol (str): Stock symbol to update

        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.needs_update(symbol):
            self.logger.info(f"Cache for {symbol} is fresh, skipping update")
            return True

        self.logger.info(f"Updating cache for {symbol}")

        # Fetch data
        hist_df = self._fetch_data(symbol)

        if hist_df is None:
            return False

        # Convert DataFrame to cache format
        cache_data = {
            'symbol': symbol,
            'last_updated': datetime.now().isoformat(),
            'days_stored': len(hist_df),
            'columns': self.columns,
            'data': {}
        }

        # Store data by date
        for date, row in hist_df.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            cache_data['data'][date_str] = {col: row[col] for col in self.columns if col in row.index}

        # Save to disk
        self._save_cache_file(symbol, cache_data)

        return True

    def update_batch(self, symbols, delay_seconds=1):
        """
        Update cache for multiple symbols with rate limiting.

        Args:
            symbols (list): List of stock symbols
            delay_seconds (float): Delay between API calls to avoid rate limiting

        Returns:
            dict: Dictionary with symbols as keys and success status as values
        """
        results = {}

        for i, symbol in enumerate(symbols):
            results[symbol] = self.update(symbol)

            # Add delay between requests (except after the last one)
            if i < len(symbols) - 1 and delay_seconds > 0:
                time.sleep(delay_seconds)

        return results

    def lookup(self, symbol, auto_update=True):
        """
        Look up cached data for a symbol.

        Args:
            symbol (str): Stock symbol
            auto_update (bool): Automatically update if cache is stale

        Returns:
            pd.DataFrame: Historical data with date index, or None if not available
        """
        # Update if needed and auto_update is enabled
        if auto_update and self.needs_update(symbol):
            self.update(symbol)

        # Load cached data
        cached_data = self._load_cache_file(symbol)

        if cached_data is None:
            return None

        # Convert to DataFrame
        data_dict = cached_data.get('data', {})

        if not data_dict:
            return None

        # Create DataFrame from cache
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        df = df.sort_index()

        return df

    def get_multiple(self, symbols, auto_update=True):
        """
        Get cached data for multiple symbols.

        Args:
            symbols (list): List of stock symbols
            auto_update (bool): Automatically update stale caches

        Returns:
            dict: Dictionary mapping symbols to their DataFrames
        """
        results = {}

        for symbol in symbols:
            df = self.lookup(symbol, auto_update=auto_update)
            if df is not None:
                results[symbol] = df

        return results

    def get_cache_info(self, symbol):
        """
        Get metadata about a cached symbol.

        Args:
            symbol (str): Stock symbol

        Returns:
            dict: Cache metadata (last_updated, days_stored, columns, is_stale)
        """
        cached_data = self._load_cache_file(symbol)

        if cached_data is None:
            return None

        info = {
            'symbol': cached_data.get('symbol'),
            'last_updated': cached_data.get('last_updated'),
            'days_stored': cached_data.get('days_stored'),
            'columns': cached_data.get('columns'),
            'is_stale': self.needs_update(symbol)
        }

        return info

    def list_cached_symbols(self):
        """
        List all symbols that have cached data.

        Returns:
            list: List of symbol strings
        """
        symbols = []

        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                symbol = filename[:-5]  # Remove .json extension
                symbols.append(symbol)

        return sorted(symbols)

    def clear_cache(self, symbol=None):
        """
        Clear cached data.

        Args:
            symbol (str, optional): Symbol to clear. If None, clears all cached data.
        """
        if symbol is None:
            # Clear all cache files
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        os.remove(filepath)
                        self.logger.info(f"Removed cache file: {filename}")
                    except OSError as e:
                        self.logger.error(f"Error removing {filename}: {e}")
        else:
            # Clear specific symbol
            cache_file = self._get_cache_file_path(symbol)
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    self.logger.info(f"Removed cache for {symbol}")
                except OSError as e:
                    self.logger.error(f"Error removing cache for {symbol}: {e}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create cache with default settings
    cache = HistoricalCache()

    # Example: Update cache for a few symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    print(f"Updating cache for {symbols}...")
    cache.update_batch(symbols)

    # Lookup data
    print("\nLooking up AAPL data:")
    aapl_data = cache.lookup('AAPL')
    if aapl_data is not None:
        print(aapl_data.tail())
        print(f"\nShape: {aapl_data.shape}")

    # Get cache info
    print("\nCache info for AAPL:")
    info = cache.get_cache_info('AAPL')
    print(info)

    # List all cached symbols
    print("\nAll cached symbols:")
    print(cache.list_cached_symbols())
