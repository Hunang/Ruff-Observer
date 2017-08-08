
# -*- coding: utf-8 -*-
# In[libraries]
# Custom libraries
import petur_functions as petur

# Base libraries
import pandas as pd
import numpy as np
import time
import operator
from collections import Counter
from datetime import datetime
from copy import copy

# Genetic Algos
from tpot import TPOTClassifier
from tpot.builtins import StackingEstimator, ZeroCount

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn import svm, linear_model, neighbors, metrics
from sklearn import cross_validation as cv

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import FastICA
from sklearn.linear_model import LogisticRegression
# In[Load data]
#==============================================================================
# # Load data 
#==============================================================================
df_train = pd.read_csv("MAIN_TRAIN_2.CSV", index_col='Date', header=0)

#==============================================================================
# ## NOTE:
#==============================================================================
# df_test is a held-out test set!!!
# SHOULD NOT BE USED, EXCEPT FOR TESTING AT KEY POINTS IN PROJECT
#==============================================================================
#==============================================================================
df_test = pd.read_csv("MAIN_TEST_2.CSV", index_col='Date', header=0)
#==============================================================================
#==============================================================================

tickers = petur.get_tickers()
tickers.append('OMXIPI')

# In[]
"""
tickers = petur.get_tickers()
tickers.append('OMXIPI')

df_train[tickers].to_csv("Benchmark_train.csv")
df_test[tickers].to_csv("Benchmark_test.csv")
"""

# In[Benchamark Scores with SVC]
days = [1, 3, 5, 10, 20]
features_all = tickers
stocks = tickers
#stocks = ['OMXIPI']
scores = {}

for stock in stocks:
    acc = []
    for day in days:
        training = df_train[features_all].copy()
        testing = df_test[features_all].copy()
        
        # Create Labels
        x_train, y_train = petur.create_better_labels(stock, training, day)
        x_test, y_test = petur.create_better_labels(stock, testing, day)
        
        # Define classifier and train
        clf = SVC(kernel = 'rbf')
        clf.fit(x_train, y_train)

        # Check accuracy and add to Scores
        accuracy = clf.score(x_test, y_test)
        name = "%s-%s" %(stock, day)
        print(name)
        acc.append(accuracy)
    scores[stock] = acc
scores = pd.DataFrame(scores)
scores = scores.transpose()
scores.columns = days
scores.to_csv("Benchmark_Scores.csv")

# In[TPOT generation]

days = [1, 3, 5, 10, 20]
features_all = tickers
stocks = tickers
#stocks = ['OMXIPI']
scores = {}

for stock in stocks:
    acc = []
    for day in days:
        name = "%s-%s" %(stock, day)
        
        training = df_train[features_all].copy()
        testing = df_test[features_all].copy()
        
        # Create Labels
        x_train, y_train = petur.create_better_labels(stock, training, day)
        x_test, y_test = petur.create_better_labels(stock, testing, day)
        
        # Define classifier and train
        print("Starting TPOT for",name)
        tpot = TPOTClassifier(generations=2, population_size=2, 
                              verbosity=2, 
                              n_jobs = -1, cv=3, max_time_mins = 20)
        tpot.fit(x_train, y_train)
        
        accuracy = tpot.score(x_test, y_test)
        print("Final score:",accuracy)
        acc.append(accuracy)
    scores[stock] = acc
scores = pd.DataFrame(scores)
scores = scores.transpose()
scores.columns = days
scores.to_csv("Benchmark_Scores.csv")

for stock in stocks:
    for day in days:
        name = "TPOT/"+name+"-"+datetime.now().strftime('%m-%d_%H-%M')
        tpot.export(name)
        print("Exported to", name)