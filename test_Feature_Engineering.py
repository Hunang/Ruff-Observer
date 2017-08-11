# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:05:41 2017

@author: petur
"""

# In[]
import pandas as pd
import petur_functions as petur

df_name = "Working_Dataset/MAIN_FULL.csv"
df = pd.read_csv(df_name, index_col='Date', header=0)

tickers = petur.get_tickers()
tickers.append('OMXIPI')
days = [1,3,5,10,20]

df = petur.make_sma(df, tickers, days)
df = petur.make_ewma(df, tickers, days)
df_train, df_test = petur.tr_ts_split(df)

print(tickers)