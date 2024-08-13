import numpy as np
import pandas as pd
from functools import partial
import yfinance as yf

def option_counts_by_date(dct):
    """
    dct is a map from expiration dates to a dataframe with a
    column Symbol that conatins sthe option symbol
    """
    out=[]
    for dt, s in dct.items():
        calls=puts=0
        for o in list(s.Symbol):
            u, d, s, t=o.split(' ')
            if t=="C":
                calls+=1
            else:
                puts +=1
        out.append((dt, calls, puts))
    df=pd.DataFrame(out, columns=["exp_date", "calls", "puts"]).set_index("exp_date")
   return df.sort_index()
    

def get_stock_closing_prices(symbol, dates):
    """
    Get closing prices for a stock on specified dates from Yahoo Finance.

    Parameters:
    - symbol: Stock symbol (e.g., 'AAPL' for Apple Inc.).
    - dates: List of date strings in the format 'YYYY-MM-DD'.

    Returns:
    - A Pandas DataFrame with Date and Closing Price columns.
    """

    # Download historical data
    stock_data = yf.download(symbol, start=min(dates), end=(pd.to_datetime(max(dates)) + pd.DateOffset(1)).strftime('%Y-%m-%d'))

    
    # Extract closing prices for specified dates
    selected_dates_data = stock_data.loc[dates]

    # Create a Pandas DataFrame
    df = pd.DataFrame({
        "Date": selected_dates_data.index,
        "Closing Price": selected_dates_data['Close'].values
    })
    
    df.set_index("Date", inplace=True)

    return df

class Prices:
    def __init__(self, symbol, start_date, end_date):
        """
        read open, high, low, close, volume from Yahoo Finance into attribute data.

        """

        # Download historical data
        self.data= yf.download(symbol, start=start_date, end=end_date)
    
    def attach_returns(self, N):
        # Calculate N-day returns and attach a new column
        if N > 0:
            self.data[f'{N}DayReturn'] = self.data['Close'].pct_change(N) * 100
        else:
            self.data[f'%dDayFwdReturn'%(-N)] = self.data['Close'].pct_change(N) * -100
            
    def add_volatility(self, N):
        # Ensure the DataFrame has the necessary columns
        
        # Calculate volatility based on the last N days of 1-day returns
        self.data[f'Volatility'] = self.data['Close'].pct_change().rolling(window=N).apply(lambda x: (x*x).mean()**0.5, raw=True)* 100*np.sqrt(252)
        
    def add_vix(self):
        # Fetch historical VIX data
        vix_data = yf.download('^VIX', start=self.data.index[0], end=self.data.index[-1])
        vix_data.rename(columns={'Close': 'VIX'}, inplace=True)
        
        # Merge VIX data with existing DataFrame
        self.data = pd.merge(self.data, vix_data[['VIX']], left_index=True, right_index=True, how='left')
        
    def add_high(self, N):
        if 'High' not in self.data.columns:
            raise ValueError("DataFrame must have a 'High' column.")
        
        if N >= 0:
            # Compute the max high over the past N days including today
            self.data[f'{N}DayHigh'] = self.data['High'].rolling(window=N, min_periods=1).max()
        else:
            # Compute the max high over the next N days including today
            self.data[f'{abs(N)}DayHighFwd'] = self.data['High'].rolling(window=abs(N), min_periods=1).max()

    def add_low(self, N):
        if 'Low' not in self.data.columns:
            raise ValueError("DataFrame must have a 'Low' column.")
        
        if N >= 0:
            # Compute the min low over the past N days including today
            self.data[f'{N}DayLow'] = self.data['Low'].rolling(window=N, min_periods=1).min()
        else:
            # Compute the min low over the next N days including today
            self.data[f'{abs(N)}DayLowFwd'] = self.data['Low'].rolling(window=abs(N), min_periods=1).min()

    def add_range_pct(self, N):
        if 'Close' not in self.data.columns:
            raise ValueError("DataFrame must have a 'Close' column.")
        
        if N >= 0:
            # Compute N-day max and min of close prices over the past N days
            mx = self.data['Close'].rolling(window=N, min_periods=1).max()
            mn = self.data['Close'].rolling(window=N, min_periods=1).min()
        else:
            # Compute N-day max and min of close prices over the next N days
            mx = self.data['Close'][::-1].rolling(window=abs(N), min_periods=1).max()[::-1]
            mn = self.data['Close'][::-1].rolling(window=abs(N), min_periods=1).min()[::-1]

        # Calculate (mx - mn) / Close and express as a percentage
        column=f'{N}DayRangePct' if N > 0 else f'{-N}DayRangePctFwd'
        self.data[column] = ((mx - mn) / self.data['Close']) * 100

def forward_return_(N, prices):
    return prices['Close'].pct_change(-N) * -100

forward_return=lambda N: partial(forward_return_, N)

class TwoVars:
    def __init__(self, X, bins=None, percentiles=[], num_bins=5, context=None):
        self.bins=bins
        self.percentiles=percentiles
        self.context=context
        if type(X) is str:
            self.X=context[X]
        else:
            self.X=X(context)
        self.set_bins(bins, num_bins)
        self.result_df = pd.DataFrame()
        self.Ys=[]
        
    def set_bins(self, bins=None, num_bins=5):
        if bins is not None:
            bin_labels = [f'{bins[i]:.1f} - {bins[i+1]:.1f}' for i in range(len(bins)-1)]
            self.bins = pd.cut(self.X, bins=bins, labels=bin_labels, precision=1)
        else:
            self.bins = pd.qcut(self.X, q=num_bins, precision=1)
        
    @staticmethod
    def ptile(percentile):
        def f(x):
            x=x.dropna()
            return np.percentile(x, percentile) if len(x) > 0 else np.NaN
        return f
    
    def conditional_expectation(self, Y, Y_name=None):
        
        if type(Y) is str:
            y=self.context[Y].groupby(self.bins)
            Y_name=Y
        else:
            y=Y(self.context).groupby(self.bins)
            n=len(self.Ys)
            Y_name=f"Y{n}" if Y_name is None else Y_name
        self.Ys.append(Y_name)
        for percentile in self.percentiles:
            self.result_df[f'P{percentile}'] = y.apply(TwoVars.ptile(percentile))
        
        self.result_df[f'E[{Y_name}|X]'] = y.agg('mean')
        self.result_df['Observations'] = y.agg('count')
        return self.result_df
   
