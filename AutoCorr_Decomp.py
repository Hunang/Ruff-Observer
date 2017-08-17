# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 02:00:46 2017

@author: petur
"""
# In[Auto Correlation]
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import numpy as np
import petur_functions as petur

name_df = "Working_Dataset/MAIN_FULL.csv"
df = pd.read_csv(name_df, index_col='Date', header=0)

tickers = petur.get_tickers()
tickers = ['OMXIPI']

for stocks in tickers:
    titles='Autocorrelation of %s' % stocks
    df_temp = df.copy()
    df_temp = df_temp[stocks].pct_change().dropna()
    plot_acf(df_temp, lags=np.arange(500), title = titles)
    
# In[De-Composition]

from statsmodels.tsa.seasonal import seasonal_decompose

name_df = "Working_Dataset/MAIN_FULL.csv"
df = pd.read_csv(name_df, index_col=False, header=0)

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

series = df['OMXIPI']

res = seasonal_decompose(series, freq=20)
resplot = res.plot()