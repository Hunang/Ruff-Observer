# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:05:03 2017
@author: petur
"""
#==============================================================================
# Regression
#==============================================================================
# In[]
#==============================================================================
# # Libraries
#==============================================================================
import petur_functions as petur

import numpy as np
import pandas as pd
import time
from scipy.stats import itemfreq

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.metrics
from sklearn import preprocessing 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

# In[]
#==============================================================================
# Load data
#==============================================================================
#Define testfile
test_file = 'Dataset/INDEX/ICEX Main (OMXIPI).xlsx'

# Create dataframe
df = pd.read_excel(test_file)

# In[4]:
#==============================================================================
# PreProcessing
#==============================================================================

# Convert timestamp to datetime
df['Date'] = pd.to_datetime(df['Date']).apply(lambda x: x.date())
df = df.sort('Date', ascending=True)

# Create return
df['Return']        = df['Price'].pct_change(periods=1)

# Create shifted price and return (% returns from yesterday)
"""
            Date   Price    Return  Return_shift
1806  2010-03-16  549.73 -0.004509      0.010504
1805  2010-03-17  556.05  0.011497     -0.004509
1804  2010-03-18  564.50  0.015196      0.011497

"""
df['Return_shift']  = df.Return.shift(1)
df['Price_shift']   = df.Price.shift(1)

# Returns are based on the last day's return and the time
# So Return3 for t0 is %change from t-4 to t-1
# Same applies for averages

df['Return3']       = df['Price_shift'].pct_change(periods=3)
df['Return5']       = df['Price_shift'].pct_change(periods=5)
df['Return10']      = df['Price_shift'].pct_change(periods=10)
df['Return20']      = df['Price_shift'].pct_change(periods=20)
df['Return50']      = df['Price_shift'].pct_change(periods=50)

# Create moving price averages
df['3DaySMA']       = df['Price_shift'].rolling(center=False,window =3).mean()
df['5DaySMA']       = df['Price_shift'].rolling(center=False,window =5).mean()
df['10DaySMA']      = df['Price_shift'].rolling(center=False,window=10).mean()
df['20DaySMA']      = df['Price_shift'].rolling(center=False,window=20).mean()
df['50DaySMA']      = df['Price_shift'].rolling(center=False,window=50).mean()

# Exponential Weighted Moving Average
df['EWMA'] = df.Price_shift.ewm(com=0.5,min_periods=0,adjust=True,ignore_na=False).mean()

# Drop NAN
# TODO: Find better approach
df = df.dropna()

# In[]
#==============================================================================
# Define features to be used in regression
#==============================================================================

df_new = df.copy()

# Define training features and target variable
features = ['High', 'Low','Open']
X = preprocessing.scale(df_new[features])    
y = df_new.Price

# Take first 90% as the train data
n_split = int(len(df)*0.9)

# Define training and testing
X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]
test_date = df_new.Date[n_split:]

print(len(X), "Total datapoints")
print(len(X_train), "Training datapoints")
print(len(X_test), " Testing datapoints")
print(len(df_new[features].columns), "Features")
print(itemfreq(df_new.Return))


# In[]
#==============================================================================
# Regression
#==============================================================================

time_Start = time.time()

# Classifier
clf = SVR(kernel = 'rbf')
clf = clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)
#petur.print_evaluation(y_test, prediction)

time_End = time.time()
print("Seconds to run:", time_End-time_Start )

# In[]


# In[37]:
#==============================================================================
# Plots
#==============================================================================
# Define time series data
real_price = pd.Series(y_test)
test_price = pd.Series(prediction)

# Plot TS
real_price.plot()
test_price.plot()
plt.legend(['Price', 'Prediction'])
plt.title('Time series', fontsize=15)
plt.show()

# In[]
# Define time series data
real_price = pd.Series(y_test)
#test_price = pd.Series(prediction)

# Plot TS
real_price.plot()
#test_price.plot()
#plt.legend(['Price', 'Prediction'])
plt.title('Time series', fontsize=15)
plt.show()