
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
#df_train = pd.read_csv("Working_Dataset/MAIN_TRAIN_2.CSV", index_col='Date', header=0)
#df_train = pd.read_csv("Working_Dataset/Benchmark_train.csv", index_col='Date', header=0)

#==============================================================================
# ## NOTE:
#==============================================================================
# df_test is a held-out test set!!!
# SHOULD NOT BE USED, EXCEPT FOR TESTING AT KEY POINTS IN PROJECT
#==============================================================================
#==============================================================================
#df_test = pd.read_csv("Working_Dataset/MAIN_TEST_2.CSV", index_col='Date', header=0)
#df_test = pd.read_csv("Working_Dataset/Benchmark_test.csv", index_col='Date', header=0)
#==============================================================================
#==============================================================================
tickers = petur.get_tickers()
tickers.append('OMXIPI')
df = pd.read_csv("Working_Dataset/MAIN_FULL.CSV", index_col='Date', header=0)
df_train, df_test = petur.tr_ts_split(df)

# In[Benchamark Scores with SVC]
time_Start =time.time()

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
    
# Save to dataframe
scores = pd.DataFrame(scores)
scores = scores.transpose()
scores.columns = days

# Timing
time_End = time.time()
elapsed = time_End-time_Start

# Save to excel
excel_name = "Scores/%i_Benchmark_Score (%i sec).csv" %(time_Start, elapsed)
scores.to_csv(excel_name)
