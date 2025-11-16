import re
import datetime
import numpy as np
import pandas as pd
import itertools
from utils import get_stock_closing_prices
import yfinance as yf

INFO=3
WARN=2
ERROR=1
SILENT=0

LOG_LEVEL=ERROR

# Trading and option constants
OPTION_CONTRACT_MULTIPLIER = 100  # Standard option contract multiplier
TRADING_DAYS_PER_YEAR = 252       # Standard trading days in a year

def set_log_level(level):
    """
    Set the log level to the specified level.

    Parameters:
    level (int): The log level to be set.

    Returns:
    None
    """
    global LOG_LEVEL
    LOG_LEVEL=level

def _log(level, *msg):
    """
    Internal logging function that checks the log level and prints messages.

    Parameters:
    level (int): The log level for this message.
    *msg: The messages to be logged.
    """
    global LOG_LEVEL
    if LOG_LEVEL >= level:
        print(*msg)

def log_info(*msg):
    """
    Log an info message if the log level is INFO or higher.

    Parameters:
    *msg: The messages to be logged.
    """
    _log(INFO, *msg)

def log_warn(*msg):
    """
    Log a warning message if the log level is WARN or higher.

    Parameters:
    *msg: The messages to be logged.
    """
    _log(WARN, *msg)

def log_error(*msg):
    """
    Log an error message if the log level is ERROR or higher.

    Parameters:
    *msg: Variable length argument list of messages to log.

    Returns:
    None
    """
    _log(ERROR, *msg)

    
def parse_option(option_str):
    """
    Parse the given option string using a regular expression to extract stock name, expiration date, strike price, and option type.
    
    Parameters:
    option_str (str): The option string to be parsed.
    
    Returns:
    Tuple[str, str, float, str]: A tuple containing stock name, expiration date, strike price, and option type.
    """
    option_regex = r'(\w+)\s+([0-9\/]+) ([0-9]+[0-9\.]*) ([CP])'
    m = re.search(option_regex, option_str)
    if m:
        stock_name = m.group(1)
        expiration = m.group(2)
        strike = float(m.group(3))
        option_type = m.group(4)
        return stock_name, expiration, strike, option_type
    else:
        return ("", "", 0, "")
    
def is_option(option_str):
    """
    Check if an input string represents a valid option.

    Args:
        option_str (str): The string to check.

    Returns:
        bool: True if the input string matches the expected format for an option,
            False otherwise.

    Raises:
        None.
    """
    if type(option_str) is not str: return False
    option_str = option_str.strip()
    m = re.match(r'^(\S+)\s+(\d{2}/\d{2}/\d{4})\s+(\d+(?:\.\d{1,2})?)\s+(C|P)$', option_str)
    return (m!=None)

def flatten(list_of_lists):
    """
    Flatten a list of lists into a single list.

    Args:
        list_of_lists (list): A list of lists to be flattened.

    Returns:
        list: A single list containing all the elements of the input lists, in the
            order they appear in the input.

    Raises:
        None.
        
    """
    return list(itertools.chain.from_iterable(list_of_lists))

def skip_rows_until(filename, strng):
    """
    Function to skip rows in a file until a specific string is found.
    
    Parameters:
    filename (str): The name of the file to be read.
    strng (str): The string to search for in the file.
    
    Returns:
    int: The number of rows skipped until the specified string is found.
    """
    with open(filename) as f:
        rows_skipped=0
        while not f.readline().startswith(strng):
            rows_skipped +=1
        return rows_skipped


def read_trades_file(filename):
    """
    Read a trades file, extract relevant columns, filter specific actions and symbols, and return the resulting dataframe and unique symbols.

    Parameters:
    filename (str): The path to the trades file.

    Returns:
    pd.DataFrame: The filtered dataframe containing 'Date', 'Action', 'Symbol', 'Quantity', 'Price' columns.
    set: A set of unique symbols extracted from the 'Symbol' column.
    """
    rows_to_skip=skip_rows_until(filename, "\"Date\"")
    
    df = pd.read_csv(filename, skiprows=rows_to_skip, skipfooter=1, engine='python')
    df = df.loc[:, ['Date', 'Action', 'Symbol', 'Quantity', 'Price']]
    df = df.sort_values(by=['Date'])
    actions=["Buy", "Sell", "Buy to Open", "Sell to Open", "Buy to Close", "Sell to Close",
            "Expired", "Assigned"]
    df=df[df.Action.isin(actions)]
    df=df[df['Symbol'].apply(is_option)].copy()
    symbols=set(df['Symbol'].str.extract(r'([A-Z]+)', expand=False))
    df["Quantity"]=df["Quantity"].astype(int)
    return df, symbols

    
def group_option_pnls_by_stock(option_pnl_dict):
    """
    Calculate the sum of PnLs for each stock from a dictionary mapping option symbols to their PnLs.

    Parameters:
    - option_pnl_dict: Dictionary mapping option symbols to their PnLs.

    Returns:
    - DataFrame with columns 'StockName' and 'SumPnL'.
    """
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(option_pnl_dict.items()), columns=['OptionSymbol', 'PnL'])

    # Extract stock name from the OptionSymbol
    df['StockName'] = df['OptionSymbol'].apply(lambda x: x.split()[0])

    # Group by StockName and calculate the sum of PnLs
    sum_pnl_by_stock = df.groupby('StockName')['PnL'].sum().reset_index()

    return sum_pnl_by_stock, sum_pnl_by_stock.PnL.sum()


def get_assigned_prices(symbol, dates):
    """
    Get the closing prices on the date prior to each designated date.

    Parameters:
    - symbol: Stock symbol (e.g., 'AAPL' for Apple Inc.).
    - dates: List of date strings in the format 'MM-DD-YYYY'.

    Returns:
    - A dictionary indexed by the designated dates with corresponding previous close prices.
    """
    if len(dates) == 0: return {}
    # Convert the dates to the correct format for yfinance
    yf_dates = [pd.to_datetime(date, format='%m-%d-%Y').strftime('%Y-%m-%d') for date in dates]

    # Download historical data
    start_date = (pd.to_datetime(min(yf_dates)) - pd.DateOffset(5)).strftime('%Y-%m-%d')
    end_date = (pd.to_datetime(max(yf_dates)) + pd.DateOffset(5)).strftime('%Y-%m-%d')
    
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
    stock_data.columns = stock_data.columns.droplevel(1)
    # Create a dictionary with original dates
    previous_close_prices_dict = {}
    for date in dates:
        formatted_date = pd.to_datetime(date, format='%m-%d-%Y').strftime('%Y-%m-%d')

        # Find the closest available trading day before the specified date
        previous_trading_day = stock_data[stock_data.index <= formatted_date].index.max()

        if not pd.isnull(previous_trading_day):
            previous_close_price = stock_data.loc[previous_trading_day, 'Close']
            previous_close_prices_dict[date] = previous_close_price
        else:
            previous_close_prices_dict[date] = None
    return previous_close_prices_dict

class Trade:
    """
    A class to represent a collection of trades.

    Attributes
    ----------
    df : pandas.DataFrame
        The DataFrame containing all the trades.
    symbols : set
        The set of all symbols in the trades.

    Methods
    -------
    get_trades(symbol)
        Returns a DataFrame of all trades for a given symbol.
    """

    def __init__(self, filename,
                exp_low=datetime.date(2000, 1, 1),
                exp_high=datetime.date(2100, 1, 1)):
        """
        Parameters
        ----------
        filename : str
            The name of the CSV file containing the trades.
        """
        self.df, self.symbols=read_trades_file(filename)
        self.exp_low=exp_low
        self.exp_high=exp_high


    def get_trades(self, symbol=None):
        """
        Returns a DataFrame of all trades for a given symbol.

        Parameters
        ----------
        symbol : str
            The symbol to get trades for.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing all trades for the specified symbol.
        """
        
        
        for i, sym in enumerate(self.df.Symbol):
            if type(sym) != str:
                log_error(i, sym, self.df.iloc[i])
                raise ValueError(f"Invalid symbol type at index {i}: {sym}")
        
        if symbol is not None:
            df = self.df[self.df['Symbol'].str.startswith(symbol + " ")] #option symbols only
        else:
            df=self.df
        df = df.loc[:, ['Date', 'Action', 'Symbol', 'Quantity', 'Price']] # select specific columns

        
        df["Price"]=df["Price"].str.strip("$").astype(float)
        # Extract date substring from 'Date' column
        df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split('as of ')[-1]))
        
        # Sort by 'Date'
        df = df.sort_values(by='Date')
        
        # Set price to 0 for expired actions
        df.loc[df['Action'] == 'Expired', 'Price'] = 0
        df.loc[df['Action'] =="Sell", 'Quantity'] *= -1
        df.loc[df['Action'].isin(['Sell to Open', 'Sell to Close']), 'Quantity'] *= -1
        
        exp_date=df.Symbol.apply(lambda s: pd.to_datetime(parse_option(s)[1]).date())
        N=len(df)
        df=df[(exp_date >= self.exp_low) & (exp_date <= self.exp_high)]
        if len(df) > 0:
            log_info("B", self.exp_low, self.exp_high, min(list(exp_date)), N, len(df), "\n\n")
            log_info(df)
        return df.reset_index(drop=True)

        
    def compute_option_pnls(self, exp_month, exp_year, symbol=None, opt_type=""):
        """
        Calculates P&Ls for trades in a given trades file, using prices from a given prices file.

        Args:
            exp_month (int): The expiration month of months
            exp_year (int): The expiration year of months
            symbol (str, optional): The symbol to calculate P&Ls for. Defaults to None.
            opt_type (str, optional): The type of option to calculate P&Ls for. Defaults to "".

        Returns:
            dict: A dictionary containing the P&Ls for each symbol in the trades file.
            pandas.DataFrame: A DataFrame containing the matched and unmatched trades.
            pandas.DataFrame: A DataFrame containing the unmatched trades.
        """

        # Initialize dictionary to store P&Ls for each symbol
        pnl_dict = {}

        # Initialize list to store matched trades
        mtrades = []

        # Initialize DataFrame to store unmatched trades
        all_unmatched_df = pd.DataFrame()

        # Loop over each symbol in self.symbols or the provided symbol
        if symbol is None:
            symbols = self.symbols
        else:
            symbols = [symbol]
        for symbol in symbols:
            # Get trades dataframe for symbol
            trades = self.get_trades(symbol)

            # Filter trades dataframe for the given expiration month and year
            option_trades = filter_by_expiration_month(trades, exp_month, exp_year, opt_type)

            # If there are no option trades, continue to the next symbol
            if len(option_trades) == 0:
                continue

            # Get stock prices for the given symbol and option trades
            stock_prices = get_assigned_prices(symbol, option_trades[option_trades.Action == "Assigned"].Date)
           

            # Pair option trades and get unmatched trades
            matched_df, unmatched_df, matched_trades = pair_option_trades(option_trades)
      
            unmatched_df=filter_unexpired_trades(unmatched_df)

            # Concatenate unmatched trades with the existing unmatched DataFrame
            if len(unmatched_df) > 0:
                log_error(symbol, "Matched:", len(matched_df), "unmatched:", len(unmatched_df), 
                "total:", len(option_trades))
                # print(unmatched_df_)
                all_unmatched_df = pd.concat([all_unmatched_df, unmatched_df], axis=0)
                print("UNMATCHED", symbol, unmatched_df)
                # print(unmatched_df_.columns)
            # Loop over matched trades
            for open_t, close_t in matched_trades:
                # If the close trade is an assignment, update its price using the stock prices
                if close_t.Action == "Assigned":
                    close_t.Price = get_option_price_on_assignment(open_t.Symbol, close_t.Date, stock_prices)

                # Calculate the option P&L and update the P&L dictionary for the given symbol
                option_pnl = calculate_option_pnl(open_t, close_t)
                pnl_dict[open_t.Symbol] = pnl_dict.setdefault(open_t.Symbol, 0) + option_pnl

                # Append the matched trade information to the matched trades list
                mtrades.append((open_t.Symbol, open_t.Date.date(), open_t.Action, open_t.Price,
                                close_t.Date.date(), close_t.Action, close_t.Price, option_pnl))

        # Create a DataFrame from the matched trades list
        mtrades_df = pd.DataFrame(mtrades, columns=["Symbol", "open_date", "action", "price",
                                                    "close_date", "close_action", "close_price", "pnl"])

        # Return the P&L dictionary, matched trades DataFrame, and unmatched trades DataFrame
        return pnl_dict, mtrades_df, all_unmatched_df


    def show_option_trades(self, symbol, exp_month, exp_year, opt_type=""):
        _, mtrades_df, unmatched_df=self.compute_option_pnls(exp_month, exp_year, symbol=symbol, opt_type=opt_type)
        display(mtrades_df)

def filter_unexpired_trades(unmatched_df):
    if len(unmatched_df) == 0: return unmatched_df
    exp=pd.to_datetime(unmatched_df.Symbol.apply(lambda x: parse_option(x)[1]), format='%m/%d/%Y')
    
    live=exp > datetime.datetime.today()-datetime.timedelta(days=3) 
    # we might be running this on a Sunday
    # don't want expired trades from Friday to cause unmatched trades
    
    return unmatched_df[~live]

def pair_option_trades(trades_df):
    """
    Matches Buy-to-open trades with later Sell-to-close trades and Sell-to-open trades with later Buy-to-close trades.
    Any unmatched trades will be printed as diagnostic messages.
    
    Args:
    trades_df: a pandas dataframe with columns Symbol, Date, Price, Quantity, Action
    
    Returns:
    A tuple of 3 pandas dataframes: 
    1. A dataframe of matched trades 
    2. A dataframe of unmatched trades
    3. A list of paired trades 
    """
    # Sort the trades by date. This is necessary so that we can match trades in the correct order.
    trades_df = trades_df.sort_values(by='Date')
    
    # Initialize variables to keep track of open trades and matched trades
    buy_open_trades = {}  # Dictionary to store buy-to-open trades keyed by symbol
    sell_open_trades = {}  # Dictionary to store sell-to-open trades keyed by symbol
    matched_trades = []  # List to store matched trades
    
    # Log the trades dataframe for debugging purposes
    log_info(trades_df.to_string())

    # Iterate over each trade in the dataframe. We need to handle the case where a trade has a quantity greater than 1 or less than -1, which can be expanded into multiple trades.
    trades = []
    for t in list(trades_df.iterrows()):
        t_=t[1].copy()
        if t_['Quantity'] == '':
            t_['Quantity']=0
        t_['Quantity']=qty=int(t_['Quantity'])
        #t_['Quantity']=qty
        if qty > 1:
            for i in range(qty):
                t_['Quantity']=1
                trades.append(t_)
        elif qty < -1:
            for i in range(-qty):
                t_['Quantity']=-1
                trades.append(t_)
        else:
            trades.append(t_)
  

    # Iterate over each trade in the dataframe
    for trade in trades:
        symbol = trade['Symbol']
        action = trade['Action']
        quantity = int(trade['Quantity'])

        # Check if trade is a buy or sell to open trade
        if 'Open' in action:
            # Add trade to appropriate open trades dictionary
            if quantity > 0:
                buy_open_trades.setdefault(symbol, [])
                buy_open_trades[symbol].append(trade)
                #print(trade)
            else:
                sell_open_trades.setdefault(symbol, [])
                sell_open_trades[symbol].append(trade)
    for trade in trades:
        symbol = trade['Symbol']
        action = trade['Action']
        quantity = int(trade['Quantity'])
            
        # Check if trade is a sell or buy to close trade
        if 'Close' in action:
            # Determine whether this is a sell to close or buy to close trade
            if quantity > 0:
                closing_trade = trade
                opening_trades = sell_open_trades
            else:
                closing_trade = trade
                opening_trades = buy_open_trades
                #print(closing_trade, buy_open_trades)
            
            # Match closing trade to a previously opened trade
            opened_trade = None
            for ii, opening_trade in enumerate(opening_trades.get(symbol,[])):
                if opening_trade['Quantity'] == -closing_trade['Quantity']:
                    opened_trade = opening_trade
                    # print("closing:", trade)
                    # print(opening_trade)
                    break
                
            if opened_trade is None:
                log_warn(f"No matching opening trade found for closing trade: {closing_trade}")
            else:
                # Add matched trades to matched_trades list and remove from open trades dictionaries
                matched_trades.append((opened_trade, closing_trade))
                del opening_trades[symbol][ii]
                
        # Check if trade is expired or assigned
        elif 'Expired' in action or 'Assigned' in action:
            # Determine whether this is an assigned or expired trade
            if quantity > 0:
                assign_trade = trade
                open_trades = sell_open_trades

            else:
                assign_trade = trade
                open_trades = buy_open_trades
            # Match assigned/expired trade to an open trade
            opened_trade = None
            for opening_trade in open_trades.get(symbol, []):
                if opening_trade['Quantity'] == -assign_trade['Quantity']:
                    opened_trade = opening_trade
                    break
                
            if opened_trade is None:
                log_error(f"No matching open trade found for assigned/expired trade: {assign_trade.Symbol} {assign_trade.Date} {assign_trade.Action}")
            else:
                # Add matched trades to matched_trades list and remove from open trades dictionaries
                matched_trades.append((opened_trade, assign_trade))
                # del opened_trade  # Remove the matched trade from the open trades dictionary
                open_trades[symbol].remove(opened_trade)
                
    
    # Create dataframes of matched and unmatched trades
    matched_df = pd.DataFrame(matched_trades, columns=['Opened Trade', 'Closing Trade'])
    # convert buy_open_trades dictionary containing trades objects to a dataframe
    buy_open_trades_df = pd.DataFrame.from_records([t.to_dict() for ts in buy_open_trades.values() for t in ts ]) 

    # convert sell_open_trades dictionary containing trades objects to a dataframe
    sell_open_trades_df = pd.DataFrame.from_records([t.to_dict() for ts in sell_open_trades.values() for t in ts ])
    unmatched_df=pd.concat([buy_open_trades_df, sell_open_trades_df])
   

    log_info(unmatched_df.to_string())
    
    
    return (matched_df, unmatched_df, matched_trades)





def filter_by_expiration_month(trades_df, expiration_month, expiration_year, opt_type):
    """
    Filter trades based on the specified expiration month and year.

    Parameters:
    - trades_df: DataFrame of trades with an 'OptionSymbol' column.
    - expiration_month: Integer representing the expiration month (1-12).
    - expiration_year: Integer representing the expiration year.

    Returns:
    - DataFrame containing only the trades with the specified expiration month and year.
    """
    if len(trades_df) ==0: return trades_df

    # Assuming 'Symbol' is the column containing option symbols in trades_df
    trades_df['ParsedOption'] = trades_df['Symbol'].apply(parse_option)

    # Extracting relevant information into separate columns
    trades_df[['StockName', 'ExpirationDate', 'Strike', 'OptionType']] = pd.DataFrame(trades_df['ParsedOption'].tolist(), index=trades_df.index)

    # Convert ExpirationDate to datetime
    trades_df['ExpirationDate'] = pd.to_datetime(trades_df['ExpirationDate'])

    # Filter trades based on the specified expiration month and year
    filtered_trades = trades_df[(trades_df['ExpirationDate'].dt.month == expiration_month) & (trades_df['ExpirationDate'].dt.year == expiration_year)]

    if (opt_type not in [None,""]):
        filtered_trades=filtered_trades[filtered_trades.OptionType == opt_type]
    # Drop temporary columns
    filtered_trades = filtered_trades.drop(columns=['ParsedOption', 'StockName', 'ExpirationDate', 'Strike', 'OptionType'])

    return filtered_trades

def calculate_option_pnl(open_trade, close_trade):
    """
    Calculates the P&L of a pair of option trades.

    Args:
    open_trade (pd.Series): An open trade for an option contract.
    close_trade (pd.Series): A closing trade for the same option contract.
    prices (dict): A dictionary mapping symbol to prices.

    Returns:
    float: The P&L of the option trade pair.
    """
    if open_trade['Symbol'] != close_trade['Symbol']:
        raise ValueError('Option trades have different symbols.')

    option = parse_option(open_trade['Symbol'])
    if not option:
        raise ValueError('Invalid option symbol.')

    qty = open_trade['Quantity']

    open_price = open_trade['Price']
    close_price=close_trade["Price"]

    pnl = qty * (close_price - open_price) * OPTION_CONTRACT_MULTIPLIER

    return pnl



def get_option_price_on_assignment(option_symbol, assignment_date, stock_prices):
    """
    Calculate the price of an option upon assignment.

    Args:
        option_symbol (str): The symbol of the option.
        assignment_date (str): The date of assignment.
        stock_prices (dict): A dictionary containing stock prices indexed by date.

    Returns:
        float: The price of the assigned option.
    """
    try:
        stock_symbol, _, strike, opt_type = parse_option(option_symbol)
        stock_price = stock_prices[assignment_date]
        assigned_price = (strike - stock_price) if opt_type == 'P' else (stock_price - strike)
        log_info("Assigned", option_symbol, strike, stock_price, assigned_price)
    except:
        log_error("assignment price failure", option_symbol, assignment_date)
        assigned_price=np.NaN
    return assigned_price


def int_(x):
    """
    A function that takes a number x and returns an integer representation of x. 
    If x is NaN, it returns 0.
    """
    if np.isnan(x): return 0
    else: return int(x)

def compute_trade_stats(pnl_dict):
    """
    Compute trade statistics based on the provided pnl_dict and return a pandas Series.
    """
    pnls=np.array(list(pnl_dict.values()))
    return pd.Series(dict(avg_loss= int_(pnls[pnls <= 0].mean()),
                avg_wins = int_(pnls[pnls > 0].mean()),
                num_loss = len(pnls[pnls <= 0]),
                num_wins = len(pnls[pnls > 0]),
                total_pnl=int_(pnls.sum())))


def summarize_options_data(df, frequency='weekly'):
    """
    Generate a summary of options trading data based on the specified frequency.
    
    Parameters:
    - df (DataFrame): The input DataFrame containing options trading data.
    - frequency (str): The frequency at which to summarize the data, can be 'weekly' or 'monthly' (default is 'weekly').
    
    Returns:
    - summary_df (DataFrame): A DataFrame summarizing the options trading data grouped by period and type.
    """
    df = df[df.Action == "Sell to Open"].copy()
    df[['Underlying', 'Expiration', 'Strike', 'Type']] = df['Symbol'].apply(parse_option).apply(lambda x: pd.Series(x[:4]))
    df['Expiration'] = pd.to_datetime(df['Expiration'], format='%m/%d/%Y')
    # Convert Quantity and Price to numeric values
    df['Quantity'] = -pd.to_numeric(df['Quantity'])
    df['Price'] = pd.to_numeric(df['Price'].replace('[$,]', '', regex=True))

    # Calculate option premium collected
    df['Premium Collected'] = df['Quantity'] * df['Price'] * OPTION_CONTRACT_MULTIPLIER

    # Group by date and type (puts or calls)
    if frequency == 'weekly':
        grouped_data = df.groupby(['Expiration', 'Type'])
    elif frequency == 'monthly':
        grouped_data = df.groupby([df['Expiration'].dt.to_period('M'), 'Type'])
    else:
        raise ValueError("Invalid frequency. Use 'weekly' or 'monthly'.")

    # Calculate the number of unique symbols, sum of positions, and total premium collected for each period and type
    summary_df = grouped_data.agg({
        'Underlying': 'nunique',
        'Quantity': 'sum',
        'Premium Collected': 'sum'
    })

    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        'Underlying': 'Unique Symbols',
        'Quantity': 'Positions',
        'Premium Collected': 'Premium Collected'
    })

    # Unstack the 'Type' level to create separate columns for puts and calls
    summary_df = summary_df.unstack('Type')

    # Convert 'Positions' and 'Premium Collected' to integers
    summary_df[('Put', 'Unique Symbols')] = summary_df[('Unique Symbols', 'P')]
    summary_df[('Put', 'Positions')] = summary_df[('Positions', 'P')].astype(int)
    summary_df[('Put', 'Premium Collected')] = summary_df[('Premium Collected', 'P')].astype(int)
    
    summary_df[('Call', 'Unique Symbols')] = summary_df[('Unique Symbols', 'C')]
    summary_df[('Call', 'Positions')] = summary_df[('Positions', 'C')].astype(int)
    summary_df[('Call', 'Premium Collected')] = summary_df[('Premium Collected', 'C')].astype(int)

    # Drop the original columns
    summary_df = summary_df.drop(columns=[('Unique Symbols', 'P'), ('Unique Symbols', 'C'),
                                        ('Positions', 'P'), ('Positions', 'C'),
                                        ('Premium Collected', 'P'), ('Premium Collected', 'C')])

    # Sort the DataFrame by the period
    summary_df = summary_df.sort_index()

    return summary_df
