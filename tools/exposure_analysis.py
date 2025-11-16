import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sys
import os

# Add project root to Python path to enable imports
# This works whether running from project root or tools directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from libs.positions_data import OptionsData
from py_vollib.black_scholes.greeks.analytical import delta
from libs.stock_cache import StockCache

# Trading and option constants
TRADING_DAYS_PER_YEAR = 252  # Standard trading days in a year

# Volatility multiplier constant (percentage premium to add to historical volatility)
# Example: 50 means 1.5x multiplier (50% premium), 100 means 2x multiplier (100% premium)
VOLATILITY_PREMIUM = 50

# Outlier capping constant for volatility calculation
# Cap outliers at K times the 90th percentile to reduce distortion from extreme returns
OUTLIER_CAP_MULTIPLIER = 4

# Maximum exposure threshold in dollars
# Used to calculate hedge needed to bring exposure within this limit
MAX_EXPOSURE = 5000

class ExposureAnalysis:
    """
    Analyzes options exposure by calculating deltas and implied volatilities.
    """
    
    def __init__(self, premium: float = 0.0):
        """
        Initialize the exposure analysis.
        
        Args:
            premium (float): Premium to add to implied volatility (default: 0.0)
        """
        self.premium = premium
        self.volatility_cache = {}  # Cache for historical volatilities
        self.stock_cache = StockCache()  # Use stock cache for API calls

    def _extract_tickers(self, df: pd.DataFrame) -> List[str]:
        """Extract unique ticker symbols from positions DataFrame."""
        if 'Ticker' in df.columns:
            # If Ticker column exists (options have been parsed), use it
            return df['Ticker'].dropna().unique().tolist()
        else:
            # Extract ticker from Symbol column
            # For options: "AAPL 11/21/2025 130.00 P" -> "AAPL"
            # For stocks: "AAPL" -> "AAPL"
            return (
                df['Symbol']
                .astype(str)
                .apply(lambda x: x.split()[0] if isinstance(x, str) and ' ' in x else x)
                .dropna()
                .unique()
                .tolist()
            )

    def _get_stock_price(self, ticker: str, stock_data: pd.DataFrame,
                         options_data: pd.DataFrame) -> Optional[float]:
        """
        Get stock price with multiple fallback strategies.

        Args:
            ticker: Stock ticker symbol
            stock_data: DataFrame containing stock positions
            options_data: DataFrame containing options positions

        Returns:
            Stock price or None if not found
        """
        # Try to get price from stock data
        if not stock_data.empty and 'Price' in stock_data.columns and stock_data['Price'].iloc[0] > 0:
            return stock_data['Price'].iloc[0]

        # Try to get price from Yahoo Finance
        try:
            yf_ticker = yf.Ticker(ticker)
            stock_info = yf_ticker.info
            if stock_info and 'regularMarketPrice' in stock_info:
                return float(stock_info['regularMarketPrice'])
        except Exception as e:
            print(f"Error getting stock price for {ticker}: {e}")

        # Use first strike price as fallback
        if not options_data.empty and 'Strike' in options_data.columns:
            stock_price = options_data['Strike'].iloc[0]
            print(f"Using strike price {stock_price} as fallback for {ticker}")
            return stock_price

        return None

    def _get_historical_volatility(self, ticker_symbol: str, days: int = 252) -> float:
        """
        Get historical volatility from cache or calculate it.
        
        Args:
            ticker_symbol (str): Stock ticker symbol
            days (int): Number of days to look back (default: 252 trading days)
            
        Returns:
            float: Historical volatility (annualized)
        """
        # Check cache first
        if ticker_symbol in self.volatility_cache:
            return self.volatility_cache[ticker_symbol]
        
        try:
            # Get volatility from stock cache
            cached_data = self.stock_cache.lookup(ticker_symbol)
            if cached_data and cached_data.get('volatility') is not None:
                # Convert percentage to decimal
                volatility = cached_data['volatility'] / 100.0
                self.volatility_cache[ticker_symbol] = volatility
                return volatility
            
            # Fallback to direct calculation if not in cache
            ticker = yf.Ticker(ticker_symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days*1.5)  # Add 50% buffer for weekends/holidays
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                print(f"No historical data available for {ticker_symbol}")
                return 0.30  # Default to 30% if no data
            
            # Calculate daily returns
            hist['Returns'] = hist['Close'].pct_change()

            # Remove NaN values
            returns = hist['Returns'].dropna()

            # Cap outliers at K times the 90th percentile
            percentile_90 = returns.abs().quantile(0.90)
            cap_threshold = percentile_90 * OUTLIER_CAP_MULTIPLIER

            # Cap returns at the threshold (both positive and negative)
            returns_capped = returns.clip(lower=-cap_threshold, upper=cap_threshold)

            # Calculate annualized volatility (standard deviation of returns * sqrt(trading days))
            volatility = returns_capped.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            # Cache the result
            self.volatility_cache[ticker_symbol] = volatility
            
            return volatility
        except Exception as e:
            print(f"Error calculating historical volatility for {ticker_symbol}: {e}")
            return 0.30  # Default to 30% if calculation fails
    
    def _get_option_delta(self, row) -> Optional[float]:
        """
        Get the delta for a specific option using Black-Scholes model.

        Args:
            row: DataFrame row containing parsed option data

        Returns:
            float: Option delta or None if not found
        """
        try:
            # Get current stock price from cache
            cached_data = self.stock_cache.lookup(row['Ticker'])
            if not cached_data or cached_data.get('price') is None:
                print(f"Could not get stock price for {row['Ticker']}")
                return None
            S = float(cached_data['price'])
            
            # Get strike price and time to expiration
            K = float(row['Strike'])
            t = (row['Expiration'] - datetime.now()).days / 365.0
            
            # Get historical volatility and add premium
            hist_vol = self._get_historical_volatility(row['Ticker'])
            #print(f"Volatility for {row['Ticker']}: {hist_vol:.4f}")
            sigma = hist_vol * (1 + self.premium / 100)  # Convert premium from percentage to decimal
            
            # Set risk-free rate (using 5% as default)
            r = 0.05
            
            # Calculate delta using Black-Scholes
            flag = 'p' if row['Option_Type'] == 'Put' else 'c'
            d = delta(flag, S, K, t, r, sigma)
            
            # The py_vollib library returns the correct sign for delta
            # We just need to multiply by 100 (standard option multiplier)
            return d * 100
            
        except Exception as e:
            print(f"Error calculating delta for {row['Symbol']}: {e}")
            print(f"Row data: {row.to_dict()}")
            return None
    
    def analyze_positions(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze options exposure for all positions.
        
        Args:
            positions_df (pd.DataFrame): DataFrame containing positions data
            
        Returns:
            pd.DataFrame: DataFrame with added options_delta and stock_delta columns
        """
        # Create copy to avoid modifying original
        df = positions_df.copy()
        
        # Initialize new columns
        df['options_delta'] = 0.0
        df['stock_delta'] = 0.0
        
        # Parse option symbols if this is an options DataFrame
        if ' ' in df['Symbol'].iloc[0]:  # Check if these are option symbols
            df = OptionsData._parse_option_symbols(df)
        
        # Pre-populate stock cache for all unique tickers
        unique_tickers = self._extract_tickers(df)
        if unique_tickers:
            print(f"Pre-populating stock cache for {len(unique_tickers)} tickers...")
            self.stock_cache.update(unique_tickers)
        
        # Process each position
        for idx, row in df.iterrows():
            # For options (has parsed data)
            if 'Ticker' in df.columns and 'Ticker' in row and row['Ticker'] is not None:
                delta = self._get_option_delta(row)
                if delta is not None:
                    # Simply multiply delta by quantity
                    # The sign will be correct because:
                    # - Puts have negative delta from Black-Scholes
                    # - Calls have positive delta from Black-Scholes
                    # - Short positions (negative quantity) will flip the sign
                    # - Long positions (positive quantity) will keep the sign
                    df.loc[idx, 'options_delta'] = delta * row['Quantity']
            else:  # For stocks
                df.loc[idx, 'stock_delta'] = row['Quantity']
        
        return df
    
    def get_total_exposure(self, positions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate total exposure by underlying symbol.
        
        Args:
            positions_df (pd.DataFrame): DataFrame containing positions data
            
        Returns:
            Dict[str, float]: Dictionary of total exposure by symbol
        """
        # Group by underlying symbol (Ticker if available, otherwise Symbol)
        exposure = {}
        
        # Get the ticker column to group by (either 'Ticker' for options or first part of 'Symbol' for stocks)
        if 'Ticker' in positions_df.columns:
            tickers = positions_df['Ticker'].fillna(positions_df['Symbol'].apply(lambda x: x.split()[0]))
        else:
            tickers = positions_df['Symbol'].apply(lambda x: x.split()[0])
        
        # Calculate exposure for each unique ticker
        for ticker in tickers.unique():
            if ticker is None:
                continue
                
            # Get all positions for this ticker
            mask = (positions_df['Ticker'] == ticker) if 'Ticker' in positions_df.columns else positions_df['Symbol'].str.startswith(ticker)
            group = positions_df[mask]
            
            # Sum options_delta and stock_delta
            total_delta = group['options_delta'].sum() + group['stock_delta'].sum()
            exposure[ticker] = total_delta
        
        return exposure

def _read_and_filter_positions(file_path: str):
    """
    Read positions from CSV file and split into stocks and options.

    Args:
        file_path: Path to positions CSV file

    Returns:
        Tuple of (options_df, stocks_df, analyzer)
    """
    from libs.positions_data import PositionsReader, StockData, OptionsData

    # Read the raw positions data
    positions_df = PositionsReader.read_positions_file(file_path)

    # Filter out "Cash & Cash Investments" row
    positions_df = positions_df[positions_df['Symbol'] != "Cash & Cash Investments"]

    # Create StockData and OptionsData instances
    stocks_data = StockData.from_positions_df(positions_df)
    options_data = OptionsData.from_positions_df(positions_df)

    return options_data.data, stocks_data.data


def _create_exposure_summary(analyzer, options_analysis, stocks_analysis):
    """
    Create exposure summary DataFrame from analyzed positions.

    Args:
        analyzer: ExposureAnalysis instance
        options_analysis: Analyzed options DataFrame
        stocks_analysis: Analyzed stocks DataFrame

    Returns:
        DataFrame with exposure summary
    """
    exposure_summary = []

    # Get unique tickers from options data only
    options_tickers = options_analysis['Ticker'].unique()

    # Track totals for summary row
    total_options_delta = 0
    total_stock_delta = 0
    total_options_exposure = 0
    total_stock_exposure = 0
    total_net_exposure = 0

    for ticker in options_tickers:
        if ticker is None:
            continue

        # Get stock data
        stock_mask = stocks_analysis['Symbol'] == ticker
        stock_data = stocks_analysis[stock_mask]
        stock_delta = stock_data['stock_delta'].sum() if not stock_data.empty else 0

        # Get options data
        option_mask = options_analysis['Ticker'] == ticker
        options_data = options_analysis[option_mask]

        # Get stock price using helper method
        stock_price = analyzer._get_stock_price(ticker, stock_data, options_data)
        stock_exposure = stock_delta * stock_price if stock_price else 0
        options_delta = options_data['options_delta'].sum() if not options_data.empty else 0

        # Get all strikes for this ticker
        strikes = sorted(options_data['Strike'].unique())
        strikes_str = ', '.join(f"{s:.2f}" for s in strikes)

        # Calculate days to expiration for each option
        today = datetime.now()
        days_to_expiration = []
        for expiry in options_data['Expiration'].unique():
            if pd.notna(expiry):
                days = (expiry - today).days
                days_to_expiration.append(days)
        days_to_expiration = sorted(set(days_to_expiration))
        days_str = ', '.join(str(d) for d in days_to_expiration)

        # Get historical volatility for this ticker
        try:
            hist_vol = analyzer._get_historical_volatility(ticker)
            avg_volatility = hist_vol * (1 + analyzer.premium / 100) * 100  # Convert to percentage
        except Exception as e:
            print(f"Error getting volatility for {ticker}: {e}")
            avg_volatility = 30.0 * (1 + analyzer.premium / 100)  # Default to 30% if calculation fails

        # Calculate options exposure using stock price
        options_exposure = options_delta * stock_price if stock_price else 0

        # Calculate net exposure
        net_exposure = stock_exposure + options_exposure

        # Calculate hedge needed (in shares) to bring exposure within MAX_EXPOSURE
        # If abs(net_exposure) > MAX_EXPOSURE, we need to hedge the excess
        # hedge_needed is negative if we need to short, positive if we need to buy
        if abs(net_exposure) > MAX_EXPOSURE and stock_price and stock_price > 0:
            # Target exposure sign matches current exposure but with magnitude = MAX_EXPOSURE
            target_exposure = MAX_EXPOSURE if net_exposure > 0 else -MAX_EXPOSURE
            excess_exposure = net_exposure - target_exposure
            hedge_needed = -excess_exposure / stock_price  # Negative sign because we hedge opposite to exposure
        else:
            hedge_needed = 0

        # Update totals
        total_options_delta += options_delta
        total_stock_delta += stock_delta
        total_options_exposure += options_exposure
        total_stock_exposure += stock_exposure
        total_net_exposure += net_exposure

        # Add row if there are options positions
        if options_delta != 0:
            exposure_summary.append({
                'Ticker': ticker,
                'options_delta': options_delta,
                'stock_delta': stock_delta,
                'stock_price': stock_price,
                'volatility': avg_volatility,
                'strikes': strikes_str,
                'days_to_expiry': days_str,
                'options_exposure': options_exposure,
                'stock_exposure': stock_exposure,
                'net_exposure': net_exposure,
                'hedge_needed': hedge_needed
            })

    # Add summary row at the top
    exposure_summary.insert(0, {
        'Ticker': 'TOTAL',
        'options_delta': total_options_delta,
        'stock_delta': total_stock_delta,
        'stock_price': None,
        'volatility': None,
        'strikes': None,
        'days_to_expiry': None,
        'options_exposure': total_options_exposure,
        'stock_exposure': total_stock_exposure,
        'net_exposure': total_net_exposure,
        'hedge_needed': None  # Not applicable for total
    })

    # Create DataFrame and sort by absolute value of net exposure (keep TOTAL at top)
    summary_df = pd.DataFrame(exposure_summary)
    total_row = summary_df.iloc[0]
    summary_df = summary_df.iloc[1:].sort_values(by='net_exposure', key=abs, ascending=False)
    summary_df = pd.concat([pd.DataFrame([total_row]), summary_df])

    return summary_df


def _display_results(summary_df, output_file='data/exposure.csv'):
    """
    Display formatted exposure analysis results and save to CSV.

    Args:
        summary_df: Exposure summary DataFrame
        output_file: Path to save the CSV file (default: 'data/exposure.csv')
    """
    # Format display
    pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', None)     # Auto-detect display width

    print("\nExposure Analysis Summary (sorted by net exposure):")
    print(summary_df.to_string(index=False))

    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


def main():
    """
    Main program to demonstrate exposure analysis functionality.
    Reads and analyzes all positions data.
    """
    try:
        # Read positions data
        print("Reading positions data...")
        options_df, stocks_df = _read_and_filter_positions('data/pos.csv')

        print(f"\nAnalyzing all positions ({len(options_df)} options, {len(stocks_df)} stocks)...")

        # Create exposure analyzer with volatility premium
        analyzer = ExposureAnalysis(premium=VOLATILITY_PREMIUM)

        # Analyze positions
        print("Analyzing options positions...")
        options_analysis = analyzer.analyze_positions(options_df)
        print("Analyzing stock positions...")
        stocks_analysis = analyzer.analyze_positions(stocks_df)

        # Create exposure summary
        print("Creating exposure summary...")
        summary_df = _create_exposure_summary(analyzer, options_analysis, stocks_analysis)

        # Display results
        _display_results(summary_df)
            
    except Exception as e:
        print(f"Error in main program: {e}")
        raise e  # Re-raise to see full traceback

if __name__ == "__main__":
    main() 