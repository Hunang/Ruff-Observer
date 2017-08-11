# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:07:08 2017

@author: petur
"""

import pandas as pd
import time

from tpot import TPOTClassifier
from sklearn.svm import SVC


import petur_functions as petur


# In[Load data and prepare]

# Load data from csv
df_name = "Working_Dataset/MAIN_FULL.csv"
df = pd.read_csv(df_name, index_col='Date', header=0)

# Define tickers to predict and days
tickers = petur.get_tickers()
tickers.append('OMXIPI')
#tickers = ['OMXIPI','MARL', 'OSSRu']

days = [1, 3, 5, 10]
sma_days = [5, 10, 20, 50]
fibbonaci = [1, 2, 3, 5, 8, 13, 21]

# Feature engineering and train
df = petur.make_sma(df, tickers, sma_days)
df = petur.make_ewma(df, tickers, sma_days)
df = df.dropna()

# Train/test split
df_train, df_test = petur.tr_ts_split(df)

# In[TPOT generation]

time_Start =time.time()

tickers = ['OMXIPI']

scores = {}
models = {}
models_full= {}

for stock in tickers:
    acc = []
    mdl = []
    mdl_full = []
    for day in days:
        time_Start_loop =time.time()
        name = "%s-%s" %(stock, day)
        
        training = df_train.copy()
        testing = df_test.copy()
        
        # Create Labels
        x_train, y_train = petur.create_better_labels(stock, training, day)
        x_test, y_test = petur.create_better_labels(stock, testing, day)
        
        # Define classifier and train
        print("Starting TPOT for",name)
        
        tpot = TPOTClassifier(generations       =100, 
                              population_size   =100,
                              offspring_size    =100,  
                              verbosity         =2, 
                              n_jobs            =-1, 
                              max_time_mins     = 75)#, cv=5)
        tpot.fit(x_train, y_train)
        
        # Test Accuracy
        accuracy = tpot.score(x_test, y_test)
        print()
        print("Final score:",accuracy)
        acc.append(accuracy)
        
        #add full model parameters to list
        pln_name = str(tpot._optimized_pipeline)
        mdl_full.append(pln_name)   
        
        #add short model parameters to list
        to_find = '('
        index_value = pln_name.find(to_find) 
        pln_name = pln_name[0:index_value]
        mdl.append(pln_name) 

        del tpot #reset classifier
        
        #TimeTaken
        time_End_loop = time.time()
        elapsed = time_End_loop-time_Start_loop
        print("Time to run %s: %i sec / (%i min)" % (name, elapsed, (elapsed/60) ))
        print("*" *100)
        
    scores[stock] = acc
    models[stock] = mdl
    models_full[stock] = mdl_full

time_End = time.time()
elapsed = time_End-time_Start
    
# Reformat scores and models 
scores = pd.DataFrame(scores)
scores = scores.transpose()
scores.columns = days
models = pd.DataFrame(models)
models = models.transpose()
models.columns = days
mdl_full = pd.DataFrame(mdl_full)
mdl_full = mdl_full.transpose()
mdl_full.columns = days

# Export scores and models to Excel
excel_name = "Scores/%i_TPOT_Benchmark_Score (%i sec).xlsx" %(time_Start, elapsed)
writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')
scores.to_excel(writer, sheet_name='Scores')
models.to_excel(writer, sheet_name='Models')
mdl_full.to_excel(writer, sheet_name='Models_full')

writer.save()

print("*" *100)
print("Saved as %s" % excel_name)
print("Total time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))

# In[]

# In[Benchamark Scores with SVC]
time_Start =time.time()

days = [1, 3, 5, 10, 20]
stocks = tickers
scores = {}

for stock in stocks:
    acc = []
    for day in days:
        training = df_train.copy()
        testing = df_test.copy()
        
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
excel_name = "Scores/%i_SVC_Score (%i sec).csv" %(time_Start, elapsed)
scores.to_csv(excel_name)

# In[Feature selections]
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
time_Start =time.time()

days = [1]
#stocks = tickers
stocks = ['OMXIPI']
features_to = petur.get_tickers()
features_to.append(stocks)

clf = LinearSVC()
clf = RFECV(clf, step=1, cv=10, n_jobs =-1)

feature_dic = {}

for stock in stocks:
    acc = []
    for day in days:
        name = "%s-%s" %(stock, day)
        training = df_train.copy()
        training = training[features_to]
        
        x_train, y_train = petur.create_better_labels(stock, training, day)

        clf = clf.fit(x_train, y_train)
        output = clf.support_
        feature_dic[name] = output
        print(output)

# Timing
time_End = time.time()
elapsed = time_End-time_Start

# In[]

import numpy as np

test_df = df_train.copy()
#test_df = test_df[test_df[output]]
feats = test_df.columns.tolist()
feats = pd.Series(feats)

featsnew = feats.copy()
featsnew = featsnew[output]
s2 = pd.Series('OMXIPI')
featsnew = featsnew.append(s2)

print(featsnew)
# In[]
time_Start =time.time()

days = [1, 3, 5, 10, 20]
stocks = ['OMXIPI']
scores = {}

for stock in stocks:
    acc = []
    for day in days:
        training = df_train.copy()
        testing = df_test.copy()
        training = training[featsnew]
        testing = testing[featsnew]
        
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
excel_name = "Scores/%i_SVC_FeatureSelection_Score (%i sec).csv" %(time_Start, elapsed)
scores.to_csv(excel_name)

