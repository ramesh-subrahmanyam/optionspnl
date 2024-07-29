
import sys, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io
from contextlib import redirect_stdout
import logging
import yfinance as yf



# Function to get earnings dates
def get_earnings_dates(symbol, start_date, end_date, finnhub_client):
    earnings = finnhub_client.earnings_calendar(
        _from=start_date,
        to=end_date,
        symbol=symbol
    )
    return pd.DataFrame(earnings['earningsCalendar'])

def setup():
    for i in range(1, 13):
        h=open(f"earnings_dates/M{i}.csv", "w")
        h.write("symbol, date\n")
        h.close()

def store(symbol, dt):
    m=int(dt[-5:-3])
    fname=f"earnings_dates/M{m}.csv"
    print(fname, symbol, dt)
    h=open(fname, "a")
    h.write(f"{symbol},  {dt}\n")
    h.close()

def store_symbols(symbols, start_date, end_date, finnhub_client):
    for symbol in symbols:
        try:
            earnings_df = get_earnings_dates(symbol, start_date, end_date, finnhub_client)
            dts=earnings_df.date
            for dt in dts: store(symbol, dt)
        except:
            print("skipping", symbol, earnings_df)
        time.sleep(2)


def list_stocks():
    stocks = []

    # Loop through the files labeled M1 to M12
    for i in range(1, 13):
        file_name = f'earnings_dates/M{i}.csv'
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
    today = datetime.datetime.today()
    end_date = today + timedelta(days=N)
    stocks = []

    # Loop through the files labeled M1 to M12
    for i in range(1, 13):
        file_name = f'earnings_dates/M{i}.csv'
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
        file_name = f'earnings_dates/M{i}.csv'
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
    today = datetime.today()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    all_stocks = {}

    # Loop through the files labeled M1 to M12
    for i in range(1, 13):
        file_name = f'earnings_dates/M{i}.csv'
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
            
        except FileNotFoundError:
            print(f"File {file_name} not found")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    # Filter out stocks with any earnings dates within the specified range
    stocks_within_range = {stock for stock, dates in all_stocks.items() if any(today <= date <= end_date for date in dates)}

    # Get the next earnings date for each stock if it falls outside the specified range
    result = []
    for stock, dates in all_stocks.items():
        future_dates = [date for date in dates if date > end_date]
        if future_dates:
            result.append((stock, min(future_dates).date()))

    return result

def get_stocks_outside_date(end_date_str):
    today = datetime.combine(datetime.today().date(), datetime.min.time())
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    all_stocks = {}
    stocks_within_range = set()

    # Loop through the files labeled M1 to M12
    for i in range(1, 13):
        file_name = f'earnings_dates/M{i}.csv'
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
    def __init__(self, symbols):
        self.symbols = symbols
        self.df = pd.DataFrame(index=symbols)
    
    def fetch_price_data(self, start_date, end_date):
        """
        Fetch historical price data from Yahoo Finance.
        
        :param start_date: Start date for fetching data in 'YYYY-MM-DD' format.
        :param end_date: End date for fetching data in 'YYYY-MM-DD' format.
        :return: DataFrame with columns 'symbol', 'date', 'price'.
        """
        data = []
        for symbol in self.symbols:
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            stock_data['symbol'] = symbol
            stock_data.reset_index(inplace=True)
            data.append(stock_data[['symbol', 'Date', 'Close']])
        
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
        n_day_returns = price_pivot.pct_change(periods=N).fillna(0) * 100  # Convert to percentage
        
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
         
