# -*- coding: utf-8 -*-
# In[]
#==============================================================================
# Libraries
#==============================================================================
# Custom libraries
import petur_functions as petur

# Base libraries
import numpy as np
import pandas as pd
import time

# Functions
from collections import Counter

# Machine learning libraries
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn import cross_validation as cv

# In[]
#==============================================================================
# Load data
#==============================================================================
df_train = pd.read_csv("MAIN_TRAIN.CSV", index_col='Date', header=0)

df = df_train.copy()

# In[]
"""
#==============================================================================
# Machine learning linear baseline
#==============================================================================
target = 'OMXIPI'

# Define features to use
features = df.columns.tolist()
features.remove('Target')

y = df.Target
X = df[features]

clf = SVC(kernel = 'linear')

accuracy = cross_val_score(clf, X, y, cv=10) #,average='binary')
f1 = cross_val_score(clf, X, y, cv=10, scoring='f1') #, average='binary')

print("Accuracy:", sum(accuracy)/len(accuracy))
print("f1:", sum(f1)/len(f1))

# In[]
#==============================================================================
# Machine learning polynomial baseline
#==============================================================================
target = 'OMXIPI'
# Define features to use
features = df.columns.tolist()
features.remove('Target')

y = df.Target
X = df[features]

clf = SVC(kernel = 'poly')

accuracy = cross_val_score(clf, X, y, cv=10) #,average='binary')
f1 = cross_val_score(clf, X, y, cv=10, scoring='f1') #, average='binary')

print("Accuracy:", sum(accuracy)/len(accuracy))
print("f1:", sum(f1)/len(f1))
"""
# In[]
time_Start =time.time()
#==============================================================================
# Machine learning RBF baseline, reduced features
#==============================================================================
# Features will be all tickers in the Icelandic stock market
features = petur.get_tickers()
features.append('OMXIPI')

y = df.Target
X = df[features]

clf = SVC()

accuracy = cross_val_score(clf, X, y, cv=10) #,average='binary')
precision = cross_val_score(clf, X, y, cv=10, scoring='precision') #, average='binary')
recall = cross_val_score(clf, X, y, cv=10, scoring='recall') #, average='binary')
f1 = cross_val_score(clf, X, y, cv=10, scoring='f1') #, average='binary')

print("Accuracy:", sum(accuracy)/len(accuracy))
print("Precision:", sum(precision)/len(precision))
print("Recall:", sum(recall)/len(recall))
print("f1:", sum(f1)/len(f1))

time_End = time.time()
print("Seconds to run:", time_End-time_Start)
# In[]
time_Start =time.time()
#==============================================================================
# One off-prediction
#==============================================================================
X_train, X_test, y_train, y_test = \
        cv.train_test_split(X, y, test_size = 0.2)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print('True spread:     ', Counter(y_test))
print('Predicted spread:', Counter(predictions))
petur.print_evaluation(y_test,predictions)
df_predict = pd.DataFrame({'Predictions': predictions, 'True_label': y_test})

time_End = time.time()
print("Seconds to run:", time_End-time_Start)

# In[]
#==============================================================================
# Timeseries Split
#==============================================================================
df = df_train.copy()

features = petur.get_tickers()
features.append('OMXIPI')

y = df.Target
X = df[features]

clf = SVC(kernel = 'linear')

accuracy    = []
precision   = []
recall      = []
f1_score    = []

tscv = TimeSeriesSplit(n_splits=10)
for train_index, test_index in tscv.split(X):
    print("\n******************************************")
    # Define start and stop indexes for timeseries crossvalidation
    train_start = train_index.min()
    train_stop = train_index.max()
    test_start = train_stop+1
    test_stop = test_index.max()
    
    # Split data based on start and stop
    X_train, X_test = X[train_start:train_stop], X[test_start:test_stop]
    y_train, y_test = y[train_start:train_stop], y[test_start:test_stop]
    
    # Make prediction
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    # Results
    petur.print_evaluation(y_test, predictions)
    print('True spread:     ', Counter(y_test))
    print('Predicted spread:', Counter(predictions))
   

    
    