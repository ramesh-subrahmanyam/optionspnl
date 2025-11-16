import pandas as pd
import numpy as np
import re
from datetime import datetime
from stock_cache import StockCache
from utils import clean_numeric

class PositionsReader:
    """Helper class for reading and cleaning positions data."""
    
    @staticmethod
    def find_header_row(file_path):
        """Find the row containing column headers in the CSV file."""
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if 'Symbol' in line:
                    return i
        raise ValueError("Could not find header row in positions file")
    
    @staticmethod
    def read_positions_file(file_path):
        """
        Read positions from CSV file and clean the data.
        
        Args:
            file_path (str): Path to the positions CSV file
            
        Returns:
            pd.DataFrame: Cleaned positions data
        """
        # Find header row
        header_row = PositionsReader.find_header_row(file_path)
        
        # Read CSV file
        df = pd.read_csv(file_path, skiprows=header_row)
        
        # Select and rename columns
        columns = {
            'Symbol': 'Symbol',
            'Qty (Quantity)': 'Quantity',
            'Price': 'Price',
            'Margin Req (Margin Requirement)': 'Margin'
        }
        
        df = df[list(columns.keys())].rename(columns=columns)
        
        # Clean numeric columns
        for col in ['Quantity', 'Price', 'Margin']:
            df[col] = df[col].apply(clean_numeric)
        
        return df

class StockData:
    """Class for handling stock position data."""
    
    def __init__(self, df=None):
        """
        Initialize StockData with optional DataFrame.
        
        Args:
            df (pd.DataFrame, optional): DataFrame containing stock positions
        """
        self.data = df if df is not None else pd.DataFrame()
    
    @classmethod
    def from_positions_df(cls, df):
        """
        Create StockData from positions DataFrame by filtering for stocks.
        
        Args:
            df (pd.DataFrame): DataFrame containing all positions
            
        Returns:
            StockData: New instance containing only stock positions
        """
        # Filter for stock positions (exclude option symbols)
        # Match both compact format (AAPL150P0421) and expanded format (AAPL 11/21/2025 150.00 P)
        # Compact: starts with letters, then digits, then P or C
        # Expanded: contains date pattern MM/DD/YYYY followed by P or C
        compact_pattern = r'^[A-Z]+\d+[PC].*$'
        expanded_pattern = r'.*\d{2}/\d{2}/\d{4}.*[PC]$'
        mask = ~(df['Symbol'].str.match(compact_pattern) | df['Symbol'].str.contains(expanded_pattern, regex=True))
        stocks_df = df[mask].copy()
        
        # Get cache instance
        cache = StockCache()
        
        # Copy over volatility and price from cache
        # Cache will automatically update missing or expired data
        for symbol in stocks_df['Symbol'].unique():
            cache_data = cache.lookup(symbol)
            if cache_data:
                mask = stocks_df['Symbol'] == symbol
                if cache_data.get('price') is not None:
                    stocks_df.loc[mask, 'Price'] = cache_data['price']
                if cache_data.get('volatility') is not None:
                    stocks_df.loc[mask, 'Volatility'] = cache_data['volatility']
        
        return cls(stocks_df)
    
    def get_symbols(self):
        """Get list of stock symbols."""
        return self.data['Symbol'].unique().tolist()
    
    def get_total_value(self):
        """Calculate total value of stock positions."""
        return (self.data['Quantity'].abs() * self.data['Price']).sum()
    
    def get_total_margin(self):
        """Get total margin requirement for stock positions."""
        return self.data['Margin'].sum()
    
    def display(self):
        """Display stock positions in a formatted way."""
        if self.data.empty:
            print("No stock positions found.")
            return
        
        print("\nStock Positions:")
        print(f"Number of positions: {len(self.data)}")
        print(f"Total value: ${self.get_total_value():,.2f}")
        print(f"Total margin: ${self.get_total_margin():,.2f}")
        print("\nPositions:")
        print(self.data.to_string(index=False))

class OptionsData:
    """Class for handling options position data."""
    
    def __init__(self, df=None):
        """
        Initialize OptionsData with optional DataFrame.
        
        Args:
            df (pd.DataFrame, optional): DataFrame containing options positions
        """
        self.data = df if df is not None else pd.DataFrame()
    
    @classmethod
    def from_positions_df(cls, df):
        """
        Create OptionsData from positions DataFrame by filtering for options.
        
        Args:
            df (pd.DataFrame): DataFrame containing all positions
            
        Returns:
            OptionsData: New instance containing only options positions
        """
        # Filter for options positions (contains MM/DD/YYYY pattern)
        mask = df['Symbol'].str.contains(r'\d{2}/\d{2}/\d{4}.*[PC]$', regex=True)
        options_df = df[mask].copy()
        
        if not options_df.empty:
            # Parse option symbols to extract additional information
            options_df = cls._parse_option_symbols(options_df)
        
        return cls(options_df)
    
    @staticmethod
    def _parse_option_symbols(df):
        """Parse option symbols to extract strike price, expiration, and type."""
        def parse_symbol(symbol):
            try:
                # Match pattern: TICKER MM/DD/YYYY STRIKE.00 P/C
                match = re.match(r'^([A-Z]+)\s+(\d{2}/\d{2}/\d{4})\s+(\d+(?:\.\d+)?)\s+([PC])$', symbol)
                if not match:
                    return pd.Series({
                        'Ticker': None,
                        'Strike': np.nan,
                        'Option_Type': None,
                        'Expiration': None
                    })
                
                ticker, date_str, strike_str, option_type = match.groups()
                
                # Parse strike price
                strike = float(strike_str)
                
                # Parse date
                expiry = datetime.strptime(date_str, '%m/%d/%Y')
                
                return pd.Series({
                    'Ticker': ticker,
                    'Strike': strike,
                    'Option_Type': 'Put' if option_type == 'P' else 'Call',
                    'Expiration': expiry
                })
            except (ValueError, IndexError, AttributeError):
                return pd.Series({
                    'Ticker': None,
                    'Strike': np.nan,
                    'Option_Type': None,
                    'Expiration': None
                })
        
        # Parse all symbols using pandas apply
        parsed_df = df['Symbol'].apply(parse_symbol)
        
        # Add the parsed columns to the original DataFrame
        result = df.copy()
        result['Ticker'] = parsed_df['Ticker']
        result['Strike'] = parsed_df['Strike']
        result['Option_Type'] = parsed_df['Option_Type']
        result['Expiration'] = parsed_df['Expiration']
        
        return result
    
    def get_symbols(self):
        """Get list of option symbols."""
        return self.data['Symbol'].unique().tolist() if not self.data.empty else []
    
    def get_total_value(self):
        """Calculate total value of option positions."""
        return (self.data['Quantity'].abs() * self.data['Price']).sum() if not self.data.empty else 0.0
    
    def get_total_margin(self):
        """Get total margin requirement for option positions."""
        return self.data['Margin'].sum() if not self.data.empty else 0.0
    
    def display(self):
        """Display option positions in a formatted way."""
        if self.data.empty:
            print("No option positions found.")
            return
        
        print("\nOption Positions:")
        print(f"Number of positions: {len(self.data)}")
        print(f"Total value: ${self.get_total_value():,.2f}")
        print(f"Total margin: ${self.get_total_margin():,.2f}")
        print("\nPositions:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(self.data.to_string(index=False))

# Example usage
if __name__ == "__main__":
    # Read positions
    df = PositionsReader.read_positions_file('pos.csv')
    stocks = StockData.from_positions_df(df)
    options = OptionsData.from_positions_df(df)

    # Display positions
    stocks.display()
    options.display() 