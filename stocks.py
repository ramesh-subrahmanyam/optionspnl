

import pandas as pd

class StockTrades:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = end_date
        self.df = None  # Initialize df attribute as None

    def read_data(self, file_path):
        try:
            # Read the CSV file into a DataFrame
            all_data = pd.read_csv(file_path)
            
            # Convert 'Date' column to datetime format - specify format explicitly
            all_data['Date'] = pd.to_datetime(all_data['Date'], format='%m/%d/%Y')
            
            # Convert start_date and end_date to pandas Timestamp objects
            # Ensure they're just dates without time components
            start_date = pd.Timestamp(self.start_date).normalize()
            end_date = pd.Timestamp(self.end_date).normalize()
            
            # Print for debugging
            # print(f"Filtering dates between {start_date} and {end_date}")
            # print(f"DataFrame date range: {all_data['Date'].min()} to {all_data['Date'].max()}")
            
            # Create the filter mask - use explicit date objects
            mask = ((all_data['Symbol'] == self.symbol) & 
                    (all_data['Date'] >= start_date) & 
                    (all_data['Date'] <= end_date) & 
                    (all_data["Action"].isin(["Buy", "Sell", "Sell Short"])))
            # print(f"filtered DataFrame date range: {all_data[mask]['Date'].min()} to {all_data[mask]['Date'].max()}")
            
            # Process the filtered data
            self.df = all_data[mask].copy()
            
            # Check if we got any data
            if self.df.empty:
                print("No data found after filtering!")
                return
                
            # Continue with processing
            self.df.loc[:, 'qty'] = self.df.apply(
                lambda row: int(row['Quantity']) if row['Action'] == 'Buy' else -int(row['Quantity']), 
                axis=1
            )
            self.df.columns = self.df.columns.str.lower()
            self.df.loc[:, 'amount'] = self.df['amount'].replace('[$,]', '', regex=True).astype(float)
            df = self.df.sort_values(by='date', ascending=True)
            df['price'] = -df['amount']/df['qty']
            self.df = df[['date', 'price', 'qty', 'amount']].reset_index().copy()
            
        except Exception as e:
            print(f"Error reading data: {e}")
    
    def infer_positions(self, current_pos):
        """
        Infers the position at each row of the DataFrame based on the current position.
        
        Args:
            current_pos: The current position value
            
        Returns:
            None, but adds a 'position' column to self.df
        """
        if self.df is None:
            print("No data available. Please call read_data method first.")
            return
        
        sorted_df=self.df
        # Calculate cumulative quantity in chronological order
        sorted_df['running_qty'] = sorted_df['qty'].cumsum()
        
        # Get the final cumulative quantity (this is the total change in position)
        total_qty_change = sorted_df['running_qty'].iloc[-1]
        
        # Calculate initial position
        initial_position = current_pos - total_qty_change
        
        # Calculate positions
        sorted_df['position'] = sorted_df['running_qty'] + initial_position
        
        # Map the positions back to the original DataFrame
        position_map = dict(zip(sorted_df.index, sorted_df['position']))
        self.df['position'] = self.df.index.map(position_map)
        self.df['starting_position'] = self.df.position - self.df.qty
        self.df['value']=self.df.position*self.df.price
        
    def infer_pnl(self, current_pos, current_price):
        """
        Calculates the profit and loss (PnL) from each row to the present time.
        PnL is calculated from the point after the transaction at the row is completed.
        
        Args:
            current_pos: The current position
            current_price: The current price
            
        Returns:
            None, but adds a 'pnl_from_here' column to self.df
        """
        if self.df is None:
            print("No data available. Please call read_data method first.")
            return
        
        # Ensure we have position information
        if 'position' not in self.df.columns:
            self.infer_positions(current_pos)
        
        # Calculate PnL for each row
        pnl_values = []
        self.df.future_cash_flows=0
        for idx in self.df.index:
            # Position after this transaction is completed
            position_after_txn = self.df.loc[idx, 'position']
            
               
            # Future transactions: later date, or same date but higher index
            current_date = self.df.loc[idx, 'date']
            future_mask = ((self.df['date'] > current_date) | 
                        ((self.df['date'] == current_date) & (self.df.index > idx)))
            
            # Sum of future cash flows (not including this transaction)
            future_cash_flows = self.df.loc[future_mask, 'amount'].sum() if any(future_mask) else 0
            self.df.loc[idx, 'future_cash_flows']=future_cash_flows
            # End state value: Current position value
            end_value = current_pos * current_price
            
            # PnL = (Current position value) + (Future cash flows) - (Position after transaction * Current price)
            pnl = end_value + future_cash_flows - (position_after_txn * self.df.loc[idx, 
            'price'])
            
            pnl_values.append(pnl)
        
        # Add PnL column to the DataFrame
        self.df['pnl_from_here'] = pnl_values
        
    def find_pnl_from_zero_pos_after_start_date(self, current_pos, current_price, start_date):
        """
        Finds PnL from the earliest point on or after start_date where position is zero.
        
        Args:
            current_pos: The current position
            current_price: The current price
            start_date: The starting date to begin looking for zero positions
            
        Returns:
            A dictionary mapping dates to PnL values and the earliest date considered
        """
        if self.df is None:
            print("No data available. Please call read_data method first.")
            return {}, None
        
        # Convert start_date to datetime if it's not already
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date)
        
        # Ensure we have position and PnL information
        if 'position' not in self.df.columns:
            self.infer_positions(current_pos)
        
        if 'pnl_from_here' not in self.df.columns:
            self.infer_pnl(current_price)
        
        # Filter for rows on or after start_date
        eligible_df = self.df[self.df['date'] >= start_date]
        
        if eligible_df.empty:
            print(f"No data available on or after {start_date}")
            return {}, None
        
        # Find rows where position is exactly zero
        zero_pos_df = eligible_df[eligible_df['starting_position'] == 0]
        
        if zero_pos_df.empty:
            print(f"No zero position found on or after {start_date}")
            return {}, eligible_df['date'].min()
        
        # Get the earliest date where position is zero
        earliest_zero_date = zero_pos_df['date'].min()
        
        # Find all transactions on that date with zero position
        zero_pos_entries = zero_pos_df[zero_pos_df['date'] == earliest_zero_date]
        
        # Create output dictionary with date to PnL mapping
        out = dict(zip(zero_pos_entries['date'], zero_pos_entries['pnl_from_here']))
        
        return out, eligible_df['date'].min()
    
    def find_pnl_from_start_date(self, current_pos, current_price, start_date):
        """
        Finds PnL from the earliest point on or after start_date where position is zero.
        
        Args:
            current_pos: The current position
            current_price: The current price
            start_date: The starting date to begin looking for zero positions
            
        Returns:
            A dictionary mapping dates to PnL values and the earliest date considered
        """
        if self.df is None:
            print("No data available. Please call read_data method first.")
            return {}, None
        
        # Convert start_date to datetime if it's not already
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date)
        
        # Ensure we have position and PnL information
        if 'position' not in self.df.columns:
            self.infer_positions(current_pos)
        
        if 'pnl_from_here' not in self.df.columns:
            self.infer_pnl(current_price)
        
        # Filter for rows on or after start_date
        eligible_df = self.df[self.df['date'] >= start_date]
        
        if eligible_df.empty:
            print(f"No data available on or after {start_date}")
            return {}, None
        return  eligible_df.iloc[0]['date'], eligible_df.iloc[0]['pnl_from_here']

    
    def compute_forward_pnl(self, current_pos, current_price):
        """
        Computes the PnL from each unique date in the DataFrame forward to the present time.
        For each date, finds the earliest subsequent date where position is zero,
        then calculates PnL from that point forward.
        
        Args:
            current_pos: The current position
            current_price: The current price
            
        Returns:
            A pandas Series indexed by date containing the PnL values
        """
        if self.df is None:
            print("No data available. Please call read_data method first.")
            return pd.Series()
        
        # Ensure we have position and PnL information
        if 'position' not in self.df.columns:
            self.infer_positions(current_pos)
        
        if 'pnl_from_here' not in self.df.columns:
            self.infer_pnl(current_price)
        
        # Get unique dates in the DataFrame
        unique_dates = sorted(self.df['date'].unique())
        
        # Initialize dictionary to store results
        results = {}
        
        for date in unique_dates:
            # Find rows on or after this date
            mask = self.df['date'] >= date
            eligible_rows = self.df[mask]
            
            if eligible_rows.empty:
                continue
            
            # Find zero position rows in the eligible rows
            zero_pos_rows = eligible_rows[eligible_rows['position'] == 0]
            
            if zero_pos_rows.empty:
                continue
            
            # Get the earliest date with a zero position
            earliest_zero_date = zero_pos_rows['date'].min()
            
            # Get all zero position rows on that earliest date
            earliest_zero_rows = zero_pos_rows[zero_pos_rows['date'] == earliest_zero_date]
            
            # Calculate the average PnL from these positions
            avg_pnl = earliest_zero_rows['pnl_from_here'].mean()
            
            # Store in results
            results[date] = avg_pnl
        
        # Convert to Series and return
        return pd.Series(results)