# -*- coding: utf-8 -*-
# In[]
import petur_functions as petur
import join_all_data as load
import pandas as pd
import numpy as np
#import sentdex_preprocessing as sentdex

# In[]
#==============================================================================
# # Load data and join dataframes
#==============================================================================
df_stocks   = load.get_price_df(load.get_market_Data())
df_stocks_new= df_stocks.dropna(how='all')

df_index    = load.get_INDEX_Data()
df_fx       = load.get_FX_Data()

tickers = petur.get_tickers()

df = df_stocks.join(df_index, how = 'left')
df = df.join(df_fx, how='left')

# In[]
#==============================================================================
# Clean data
#==============================================================================
df = load.get_price_df(df)                          # Price only
df = df.fillna(0)                                   # Replace nan with 0
df = df[df.index.dayofweek < 5]                     # Remove non-working days
df.rename(columns=lambda x: x[:-6], inplace=True)    # Remove '_Price' from names
df = load.get_cleaned_df(df)                        # 1jan'10 - 30dec'16

# In[]
target = 'OMXIPI'
tickers, df = petur.create_labels(target, df)

df_ml = df.copy()

# In[]
#==============================================================================
# Machine Learning 
#==============================================================================
from collections import Counter
from sklearn import cross_validation as cv
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

df = df_ml.copy()

features = df.columns.tolist().remove('Target')

y = df.Target
X = df[features]
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, 
                                                       test_size = 0.25)

classifiers_vote = [('lsvc', svm.LinearSVC()),
                    ('knn', neighbors.KNeighborsClassifier()),
                    ('rfor', RandomForestClassifier())]
clf = VotingClassifier(classifiers_vote)
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)
print("Accuracy: ", confidence)
predictions = clf.predict(X_test)
print('True spread:     ', Counter(y_test))
print('Predicted spread:', Counter(predictions))
df_predict = pd.DataFrame({'Predictions': predictions, 'True_label': y_test})