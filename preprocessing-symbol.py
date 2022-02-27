import pandas as pd
import os

all_symbol = pd.read_csv("./data/stock/list.csv")

# csv sample 
#,symbol,name,price,exchange,exchangeShortName,type
# 0,SPY,SPDR S&P 500 ETF Trust,437.75,New York Stock Exchange Arca,AMEX,etf

#print ( all_symbol['exchangeShortName'].value_counts() )
all_symbol = all_symbol.drop( all_symbol.columns[0], axis=1)
target_stock_symbol = all_symbol [ (all_symbol['type'] == "stock")  & ( (all_symbol['exchangeShortName'] == 'NASDAQ') | (all_symbol['exchangeShortName'] == 'NYSE')) ] 
target_stock_symbol = target_stock_symbol.reset_index(drop=True)
target_stock_symbol = target_stock_symbol.drop(['price', 'exchange', 'name'], axis=1)
print(target_stock_symbol)
target_stock_symbol.to_csv('./target_stock_list.csv')