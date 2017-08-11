# -*- coding: utf-8 -*-
"""
Labels
MAs
Split
Normalise training data (value/maxValue)
Normalise testing data, using maxValue from training data
Feature Selection using training data
Prediction
"""
# In[Libraries]
import pandas as pd
import petur_functions as petur
import time

from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

# In[Load data]

# Load data from csv
df_name = "Working_Dataset/MAIN_FULL.csv"
df_raw = pd.read_csv(df_name, index_col='Date', header=0)

# In[Feature Engineering]
df_fe = df_raw.copy()

# Get list of tickers
tickers = petur.get_tickers()
tickers.append('OMXIPI')
#tickers = ['OMXIPI','MARL', 'OSSRu']   #45% of market cap + index

# Create MAs
sma_days = [5, 10, 20, 50]
fibbonaci = [1, 2, 3, 5, 8, 13, 21]

df_fe = petur.make_sma(df_fe, tickers, sma_days)
df_fe = petur.make_ewma(df_fe, tickers, sma_days)
df_fe = df_fe.dropna()

# In[Test/Train split]
df_split = df_fe.copy()
df_train, df_test = petur.tr_ts_split(df_split)

# In[Normalise data]
# Store Training Data max values

# In[Feature Selection]
time_Start =time.time()

clf = LinearSVC()
clf = ExtraTreesClassifier(n_estimators=100, random_state = 0)
clf = RFECV(clf, step=1, cv=2, n_jobs =-1)
days = [1, 3]#, 5, 10, 20]    # Days to predict in the future
stocks = ['OMXIPI','MARL']#, 'OSSRu']   #45% of market cap + index

feature_dic = {}

for stock in stocks:
    acc = []
    for day in days:
        name = "%s-%s" %(stock, day)
        print("Starting feature selection for %s" % name)
        
        training = df_train.copy()
        training = training[tickers]    #just for testing, can be deleted
        features = pd.Series(training.columns.tolist())     #get the full feature names
        
        # create labels to predict
        x_train, y_train, stuff = petur.create_better_labels(stock, training, day)
#        print(stuff)
        clf = clf.fit(x_train, y_train)
        output = clf.support_
        final_features = features[output]
        feature_dic[name] = final_features
        print("Done")
        
print(feature_dic)

# Timing
time_End = time.time()
elapsed = time_End-time_Start
print("Total time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))

# In[Prediciton]

feature_dic["lol"] = 1