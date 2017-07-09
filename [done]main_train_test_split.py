# -*- coding: utf-8 -*-
# In[]
"""
## SHOULDN'T RUN THIS AGAIN
## This was a one off to generate the training and testing data
## Will be used from now on in training and testing our models
"""
"""
import petur_functions as petur
import data_manipulation as load
import pandas as pd
import numpy as np
import time
import operator
#import sentdex_preprocessing as sentdex

# In[]
#==============================================================================
# # Load data and join dataframes
#==============================================================================
time_Start = time.time()

df_stocks   = load.get_price_df(load.get_market_Data())
df_stocks_new= df_stocks.dropna(how='all')

df_index    = load.get_INDEX_Data()
df_fx       = load.get_FX_Data()

tickers = petur.get_tickers()

df = df_stocks.join(df_index, how = 'left')
df = df.join(df_fx, how='left')

df_load = df.copy()

time_End = time.time()
print("Seconds to run:", time_End-time_Start )

# In[]
#==============================================================================
# Clean data
#==============================================================================
df = df_load.copy()

df = load.get_price_df(df)                          # Price only
df = df.fillna(0)                                   # Replace nan with 0
df = df[df.index.dayofweek < 5]                     # Remove non-working days
df.rename(columns=lambda x: x[:-6], inplace=True)    # Remove '_Price' from names
df = load.get_cleaned_df(df)                        # 1jan'10 - 30dec'16

# In[]
# Replace 0 values in non-ticker columns with the value that came before it
all_columns = df.columns.tolist()
columns_to_clean = []
for column in all_columns:
    if column not in tickers:
        columns_to_clean.append(column)
        df[column] = df[column].replace(to_replace=0, method='ffill')
#print(columns_to_clean)

df_clean = df.copy()

# In[]
import petur_functions as petur
target = 'OMXIPI'
tickers, df = petur.create_labels(target, df, days=7, change=0.02, binary=True)
df_backup = df.copy()

# In[]
df_backup.to_csv("MAIN_FULL.csv")

# In[]
from sklearn.model_selection import train_test_split

## SHOULDN'T RUN THIS AGAIN
## This was a one off to generate the training and testing data
## Will be used from now on in training and testing our models

train, test = train_test_split(df_backup, test_size = 0.2)

train.sort_index(inplace = True)
test.sort_index(inplace = True)

train.to_csv("MAIN_TRAIN.csv")
test.to_csv("MAIN_TEST.csv")

"""