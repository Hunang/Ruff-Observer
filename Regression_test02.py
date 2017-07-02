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

from sklearn import preprocessing 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# In[]
#==============================================================================
# Load data
#==============================================================================
#Define testfile
test_file = 'Dataset/INDEX/ICEX Main (OMXIPI).xlsx'

# Create dataframe
df = pd.read_excel(test_file)

# Preprocess
# Create features
df = petur.index_preprocess(df)

# In[]
#==============================================================================
# Define features to be used in regression
#==============================================================================

df_new = df.copy()

# Define training features and target variable
features = df_new.columns[10:]
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
df_svr = df.copy()
features = ['High','Low','Open']
X = preprocessing.scale(df_svr[features])    
y = df_svr.Price 

# Take first 90% as the train data
n_split = int(len(df_svr)*0.9)

# Define training and testing
X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]
test_date = df_svr.Date[n_split:]


time_Start = time.time()

# Classifier
clf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) 
clf = clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)
petur.print_evaluation(y_test, prediction, "SVR")


time_End = time.time()
print("Seconds to run:", time_End-time_Start )

# In[37]:
#==============================================================================
# Plots
#==============================================================================
# Define time series data
real_price = pd.Series(y_test)
test_price = pd.Series(prediction)

petur.print_ts_vs_prediction(real_price, test_price)