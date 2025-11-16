import os
import json
import time
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Constants
TRADING_DAYS_PER_YEAR = 252  # Standard trading days in a year

class StockCache:
    """
    A cache for storing stock data from Yahoo Finance.
    Maintains price, volatility, and other market data for stocks with timestamp tracking.
    """
    
    def __init__(self, cache_file='data/stock_cache.json', cache_interval_hours=6):
        """
        Initialize the stock cache.
        
        Args:
            cache_file (str): Path to the JSON file for persistent storage
            cache_interval_hours (int): How often to update cached data in hours
        """
        self.cache_file = cache_file
        self.cache_interval = timedelta(hours=cache_interval_hours)
        self.cache_data = self._load_cache()
    
    def _load_cache(self):
        """Load the cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading cache: {e}")
        return {'last_updated': None}
    
    def _save_cache(self):
        """Save the current cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, indent=2)
            print(f"Cache updated at {datetime.now()}")
        except IOError as e:
            print(f"Error saving cache: {e}")
    
    def _calculate_volatility(self, symbol):
        """
        Calculate 30-day volatility from historical price data.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Annualized volatility as a percentage
        """
        try:
            # Get 30 days of historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Fetch historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate standard deviation of returns
            std_dev = returns.std()

            # Annualize (multiply by sqrt of trading days)
            annualized_vol = std_dev * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            # Convert to percentage
            volatility_percent = annualized_vol * 100
            
            return round(volatility_percent, 2)
            
        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return None
    
    def needs_update(self, symbol=None):
        """
        Check if the cache needs to be updated.
        
        Args:
            symbol (str, optional): Check specific symbol. If None, checks entire cache.
            
        Returns:
            bool: True if cache needs update, False otherwise
        """
        current_time = datetime.now()
        
        # If checking a specific symbol
        if symbol:
            if symbol not in self.cache_data:
                return True
            last_updated = datetime.fromisoformat(self.cache_data[symbol].get('last_updated', '1970-01-01'))
            return current_time - last_updated > self.cache_interval
        
        # Checking entire cache
        if 'last_updated' not in self.cache_data or not self.cache_data['last_updated']:
            return True
        
        last_updated = datetime.fromisoformat(self.cache_data['last_updated'])
        return current_time - last_updated > self.cache_interval
    
    def update(self, symbols):
        """
        Update cache data for given symbols from Yahoo Finance.
        Automatically handles missing symbols and expired data.
        
        Args:
            symbols (list): List of stock symbols to update
            
        Returns:
            dict: Dictionary of symbols that were successfully updated
        """
        updated_symbols = {}
        current_time = datetime.now()
        
        # Filter symbols that need updating
        symbols_to_update = [symbol for symbol in symbols if self.needs_update(symbol)]
        
        for symbol in symbols_to_update:
            try:
                print(f"Updating cache for symbol: {symbol}")
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Calculate volatility from historical data
                volatility = self._calculate_volatility(symbol)
                
                # Extract relevant data
                self.cache_data[symbol] = {
                    'price': info.get('regularMarketPrice', None),
                    'volatility': volatility,
                    'volume': info.get('regularMarketVolume', None),
                    'market_cap': info.get('marketCap', None),
                    'last_updated': current_time.isoformat()
                }
                updated_symbols[symbol] = self.cache_data[symbol]
                
                # Add delay between requests to avoid rate limiting
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                # Still add delay even on error to avoid overwhelming the API
                time.sleep(1)
        
        # Update cache timestamp
        if updated_symbols:
            self.cache_data['last_updated'] = current_time.isoformat()
            self._save_cache()
        
        return updated_symbols
    
    def lookup(self, symbol):
        """
        Look up cached data for a symbol. Automatically updates if missing or expired.
        
        Args:
            symbol (str): Stock symbol to look up
            
        Returns:
            dict: Cached data for the symbol or None if update failed
        """
        if self.needs_update(symbol):
            self.update([symbol])
        
        return self.cache_data.get(symbol)
    
    def get_all_data(self):
        """
        Get all cached data that hasn't expired.
        
        Returns:
            dict: Dictionary of all valid cached data
        """
        current_time = datetime.now()
        valid_data = {}
        
        for symbol, data in self.cache_data.items():
            if symbol != 'last_updated':
                last_updated = datetime.fromisoformat(data.get('last_updated', '1970-01-01'))
                if current_time - last_updated <= self.cache_interval:
                    valid_data[symbol] = data
        
        return valid_data
    
    def clear(self):
        """Clear all cached data."""
        self.cache_data = {'last_updated': None}
        self._save_cache()

    def _format_datetime(self, iso_string):
        """Format ISO datetime string for display."""
        return datetime.fromisoformat(iso_string).strftime('%Y-%m-%d %H:%M:%S')

    def display(self, symbols=None, format='table'):
        """
        Display cached data in a formatted way.
        
        Args:
            symbols (list, optional): List of symbols to display. If None, displays all.
            format (str): Output format, one of 'table', 'json', or 'dict'
            
        Returns:
            Formatted display of cached data
        """
        # Get data to display
        if symbols:
            data = {symbol: self.lookup(symbol) for symbol in symbols if self.lookup(symbol)}
        else:
            data = self.get_all_data()
        
        if not data:
            print("No valid cached data found.")
            return None
        
        # Format based on requested output
        if format == 'table':
            # Convert to DataFrame for nice tabular display
            rows = [
                {
                    'Symbol': symbol,
                    'Price': f"${info['price']:.2f}" if info['price'] else 'N/A',
                    'Volatility': info['volatility'] if info['volatility'] else 'N/A',
                    'Volume': f"{info['volume']:,}" if info['volume'] else 'N/A',
                    'Market Cap': f"${info['market_cap']:,.0f}" if info['market_cap'] else 'N/A',
                    'Last Updated': self._format_datetime(info['last_updated'])
                }
                for symbol, info in data.items() if info
            ]

            df = pd.DataFrame(rows)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(df.to_string(index=False))
            return df
        
        elif format == 'json':
            # Pretty print JSON
            print(json.dumps(data, indent=2))
            return data
        
        elif format == 'dict':
            # Simple dictionary display
            for symbol, info in data.items():
                print(f"\n{symbol}:")
                for key, value in info.items():
                    if key != 'last_updated':
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {self._format_datetime(value)}")
            return data
        
        else:
            print(f"Unknown format: {format}")
            return None

# Example usage:
if __name__ == "__main__":
    # Create cache instance
    cache = StockCache()
    
    # Example symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    # Update cache if needed
    if cache.needs_update():
        print("Updating cache...")
        cache.update(symbols)
    
    # Display cache in different formats
    print("\nDisplay as table:")
    cache.display(symbols, format='table')
    
    print("\nDisplay as JSON:")
    cache.display(symbols, format='json')
    
    print("\nDisplay as dictionary:")
    cache.display(symbols, format='dict') 