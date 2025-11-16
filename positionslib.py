import pandas as pd
import pnllib

COLUMNS=['Symbol', 'Description', 'Quantity', 'Price', 'Price Change %',
       'Price Change $', 'Market Value', 'Day Change %', 'Day Change $',
       'Cost Basis', 'Gain/Loss %', 'Gain/Loss $', 'Ratings',
       'Reinvest Dividends?', 'Capital Gains?', '% Of Account',
       'Security Type', 'Margin Requirement']

def get_margin(x):
    y=x.split(" ")[-1][:-1]
    try:
        return float(y)
    except:
        return 0.0


def skip_lines(in_filename, out_filename, needed_columns=None):
    h=open(out_filename, "w") # create an output file
    #convert this code into one that only saves specific columns to a csv file
    #read the first line in the input file
    
    lines=[]
    columns=COLUMNS
    with open(in_filename) as file:
        for line in file:
    
            if line.startswith("\"Symbol"):
                line=line.replace('"', '').strip()
                columns=[x for x in line.split(",") if len(x)>0]
                break
        for line in file:
            if line.startswith("\"Cash & Cash Investments"): break
            line=line[1:-1].replace("$", "").strip().split('","')
            
            lines.append(line)
    df=pd.DataFrame(lines, columns=columns)
    #print(df.head(2))
    df["Margin"]=df["Margin Req (Margin Requirement)"].apply(get_margin)
    for col in ["Cost Basis", "Gain $ (Gain/Loss $)", "Qty (Quantity)"]:
        df[col]=df[col].str.replace("N/A","0").str.replace(",","")
    

    df["PnL"]=pd.to_numeric(df["Gain $ (Gain/Loss $)"], errors='coerce').fillna(0)
    df["Quantity"]=pd.to_numeric(df["Qty (Quantity)"], errors='coerce').fillna(0).astype(int)
    print("Total margin", int(df.Margin.sum()))
    
    if needed_columns:
        df=df[needed_columns]
    df.to_csv(h, index=False)
    
    h.close()

    return lines

def read_positions_file(in_filename, needed_columns=None):
    out_filename="pos.csv"
    skip_lines(in_filename, out_filename, needed_columns)
    df=pd.read_csv(out_filename, delimiter=",")
    if needed_columns:
        df=df.loc[:,needed_columns]

    # Quantity should be an int and Cost Basis shoulkd be a float; Remove the $ sign from Cost Basis and convert to float.  
    # The $ sign could be after a minus sign
    # df["Cost Basis"]=df["Cost Basis"].str.replace("$", "").str.replace("-","-0").astype(float)
    df["Cost Basis"]=pd.to_numeric(df["Cost Basis"], errors='coerce').fillna(0)

    df["PnL"]=pd.to_numeric(df["PnL"], errors='coerce').fillna(0)
    df["Quantity"]=df["Quantity"].astype(float).astype(int)
    df=df.sort_values(by=['Symbol'])
    return df

def get_expiration_indexed_positions(filename, needed_columns):
    df=read_positions_file(filename, None)
    # read each row 
    # have a dictionary mapping expirations to a dataframe
    dct={}
    # dictionary maps each expiration to a list of rows

    for index, row in df.iterrows():
       # extract the Symbol column
        symbol=row['Symbol']
        # check if the symbol is an option and if so parse the option name
        # the function parse_option is defied in pnllib.py
        if pnllib.is_option(symbol): 
            stock_name, expiration, strike, opt_type=pnllib.parse_option(symbol)
            # store the row in the dictionary with expiration as the key
            if expiration not in dct:
                dct[expiration]=[]
            dct[expiration].append(row)
    # convert each list of rows in the dictionary to a dataframe
    for key in dct:
        dct[key]=pd.DataFrame(dct[key],
                             columns=needed_columns)
    return dct

    
    
