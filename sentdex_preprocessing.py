# -*- coding: utf-8 -*-

# https://pythonprogramming.net/targets-for-machine-learning-labels-python-programming-for-finance/?completed=/preprocessing-for-machine-learning-python-programming-for-finance/
# In[]
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn import cross_validation as cv
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import time

def process_data_for_labels(ticker, df):
    # Each model on a per company basis
    # Each comapny takes into account each of the other companies
    
    hm_days = 7 # day range, how many days are we looking at in the future
    
    # list of ticker names
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
            
    for i in range(1,hm_days+1):
        #% change values for the next 7 days
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        
    df.fillna(0, inplace=True)
    return tickers, df
    
def buy_sell_hold(*args):
    
    cols = [c for c in args]
    requirement = 0.02 
    
    for col in cols:
        if col > requirement:       # buy if above 0.02 
            return 1
        elif col < -requirement:    # sell if below 0.02
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, 
                                            df['{}_1d'.format(ticker)],
                                            df['{}_2d'.format(ticker)],
                                            df['{}_3d'.format(ticker)],
                                            df['{}_4d'.format(ticker)],
                                            df['{}_5d'.format(ticker)],
                                            df['{}_6d'.format(ticker)],
                                            df['{}_7d'.format(ticker)]))
    
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    #print('Data spread: ', Counter(str_vals))
    
    df.fillna(0, inplace=True)
    
    
    df = df.replace([np.inf, -np.inf], np.nan)      #for infinite values like change from 0 > 1 = inf
    df.dropna(inplace=True)                         #drop those

    df_vals = df[[ticker for ticker in tickers]].pct_change()   #todays value change from yesterday
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace = True)
    
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    
    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, 
                                                           test_size = 0.25)
 
    #clf = neighbors.KNeighborsClassifier()
    classifiers_vote = [('lsvc', svm.LinearSVC()),
                        ('knn', neighbors.KNeighborsClassifier()),
                        ('rfor', RandomForestClassifier())]
    clf = VotingClassifier(classifiers_vote)
    clf.fit(X_train, y_train)
    
    confidence = clf.score(X_test, y_test)
    #print("Accuracy: ", confidence)
    #predictions = clf.predict(X_test)
    #print('Predicted spread:', Counter(predictions))
    
    return confidence

#do_ml('EIK')

time_Start =time.time()
tickers, df = process_data_for_labels('EIK')

total_acc = []

for ticker in tickers:
    acc = []
    for i in range(10):
        x = do_ml(ticker)
        acc.append(x)
        total_acc.append(x)
    print(ticker, "Accuracy: ", np.mean(acc))

print("\nTotal accuracy: ", np.mean(total_acc))
print("#Predictions run: ", len(total_acc)*3)

time_End = time.time()

print("Seconds to run:", time_End-time_Start)

