import pandas as pd

class StockTrades:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.df = None  # Initialize df attribute as None

    def read_data(self, file_path):
        # Assuming file_path is the path to the CSV file containing stock market transactions data
        try:
            # Read the CSV file into a DataFrame
            all_data = pd.read_csv(file_path)
            
            # Convert 'Date' column to datetime format
            all_data['Date'] = pd.to_datetime(all_data['Date'], format='%m/%d/%Y')
            
            # Filter rows based on symbol and date range
            mask = ((all_data['Symbol'] == self.symbol) & (all_data['Date'] >= self.start_date) 
                    & (all_data['Date'] <= self.end_date) & (all_data["Action"].isin(["Buy", "Sell", "Sell Short"])))
            self.df = all_data[mask].copy()
            self.df.loc[:, 'qty'] = self.df.apply(lambda row: row['Quantity'] if row['Action'] == 'Buy' else -row['Quantity'], axis=1)
            self.df.columns = self.df.columns.str.lower()
            self.df.loc[:, 'amount'] = self.df['amount'].replace('[\$,]', '', regex=True).astype(float)

            self.df = self.df[['date', 'qty', 'amount']].reset_index().copy()
        except Exception as e:
            print(f"Error reading data: {e}")

    def find_indices(self, current_pos):
        if self.df is None:
            print("No data available. Please call read_data method first.")
            return []

        pos = 0
        result_indices = []
        positions=[]
        for index, row in self.df.iterrows():
            pos += row['qty']
            positions.append(pos)
            if pos == current_pos:
                result_indices.append(index)
        return result_indices, positions

    def find_pnl(self, current_pos, current_price):
        out=[]
        ixs, _=self.find_indices(current_pos)
        for i in ixs:
            # print(self.df.date.iloc[i], i, 
            #             self.df.qty.iloc[:i+1].sum(),
            #             self.df.amount.iloc[:i+1].sum()+current_pos*current_price)
            out.append((self.df.date.iloc[i], 
                        self.df.amount.iloc[:i+1].sum()+current_pos*current_price))  
        return dict(out)
    
    

            
   


    
