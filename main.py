# -*- coding: utf-8 -*-
# In[]
import petur_functions as petur
import data_manipulation as load
import pandas as pd
import numpy as np
import time
import operator
#import sentdex_preprocessing as sentdex

# In[]
#==============================================================================
# # Load data and join dataframes
#==============================================================================
time_Start = time.time()

df_stocks   = load.get_price_df(load.get_market_Data())
df_stocks_new= df_stocks.dropna(how='all')

df_index    = load.get_INDEX_Data()
df_fx       = load.get_FX_Data()

tickers = petur.get_tickers()

df = df_stocks.join(df_index, how = 'left')
df = df.join(df_fx, how='left')

df_load = df.copy()

time_End = time.time()
print("Seconds to run:", time_End-time_Start )

# In[]
#==============================================================================
# Clean data
#==============================================================================
df = df_load.copy()

df = load.get_price_df(df)                          # Price only
df = df.fillna(0)                                   # Replace nan with 0
df = df[df.index.dayofweek < 5]                     # Remove non-working days
df.rename(columns=lambda x: x[:-6], inplace=True)    # Remove '_Price' from names
df = load.get_cleaned_df(df)                        # 1jan'10 - 30dec'16

df_clean = df.copy()


# In[]
#==============================================================================
# Machine Learning - One off
#==============================================================================
from collections import Counter
from sklearn import cross_validation as cv
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score

time_Start =time.time()

df = df_clean.copy()

target = 'OMXIPI'
tickers, df = petur.create_labels(target, df, days=7, change=0.02, binary=True)

features = df.columns.tolist()
features.remove('Target')

y = df.Target
X = df[features]

X_train, X_test, y_train, y_test = \
        cv.train_test_split(X, y, test_size = 0.1)

classifiers_vote = [('lsvc', svm.LinearSVC()),
                    ('knn', neighbors.KNeighborsClassifier()),
                    ('rfor', RandomForestClassifier())]

clf = VotingClassifier(classifiers_vote)

#accuracy = cross_val_score(clf, X, y, cv=10) #,average='binary')
#f1 = cross_val_score(clf, X, y, cv=10, scoring='f1') #, average='binary')

print("Accuracy:", sum(accuracy)/len(accuracy))
print("f1:", sum(f1)/len(f1))

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
# Generate Genetic algorithm output
#==============================================================================
from tpot import TPOTClassifier
from datetime import datetime

time_Start =time.time()

df = df_clean.copy()

target = 'OMXIPI'
tickers, df = petur.create_labels(target, df, 
                                  days=10, 
                                  change=0.03, 
                                  binary=True )
features = df.columns.tolist()
features.remove('Target')

y = df.Target
X = df[features]

X_train, X_test, y_train, y_test = \
        cv.train_test_split(X, y, test_size = 0.2)

tpot = TPOTClassifier(generations=10, population_size=10, 
                      verbosity=2, scoring='f1', 
                      n_jobs = -1, cv=10)

tpot.fit(X_train, y_train)
print("Final score:",tpot.score(X_test, y_test))

name = "tpot "+datetime.now().strftime('%m-%d %H:%M')
name = name+".py"
tpot.export(name)

time_End = time.time()
print("Seconds to run:", time_End-time_Start )
# In[]

# In[]
#==============================================================================
# Implement Genetic Algorithm
#==============================================================================
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

time_Start =time.time()
# Load dataframe and create labels
df = df_clean.copy()
target = 'OMXIPI'
tickers, df = petur.create_labels(target, df, days=10, change=0.025, binary=True)

# Define features
features = df.columns.tolist()
features.remove('Target')
y = df.Target
X = df[features]
X_train, X_test, y_train, y_test = \
    cv.train_test_split(X, y, test_size = 0.25)

clf1 = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.8500000000000001, n_estimators=100)),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=10, max_features=0.8, min_samples_leaf=17, min_samples_split=4, subsample=0.7500000000000001)
)

clf2 = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=2, min_samples_split=3, n_estimators=100)),
    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=5, n_estimators=100)
)

clf3 = make_pipeline(
    PolynomialFeatures(interaction_only=False),
    ExtraTreesClassifier(criterion="entropy", max_features=0.1, min_samples_split=3, n_estimators=100)
)

classifiers_vote = [('1', clf1),
                    ('2', clf2),
                    ('3', clf3)]

clf = VotingClassifier(classifiers_vote)

#accuracy = cross_val_score(clf, X, y, cv=5)
#f1 = cross_val_score(clf, X, y, cv=5, scoring='f1')

print("Accuracy:", sum(accuracy)/len(accuracy))
print("f1:", sum(f1)/len(f1))

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('True spread:     ', Counter(y_test))
print('Predicted spread:', Counter(predictions))
petur.print_evaluation(y_test,predictions)
df_predict = pd.DataFrame({'Predictions': predictions, 'True_label': y_test})


time_End = time.time()
print("Seconds to run:", time_End-time_Start )

# In[]

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator

time_Start =time.time()
# Load dataframe and create labels
df = df_clean.copy()
target = 'OMXIPI'
tickers, df = petur.create_labels(target, df, days=10, change=0.025, binary=True)

# Define features
features = df.columns.tolist()
features.remove('Target')
y = df.Target
X = df[features]
X_train, X_test, y_train, y_test = \
    cv.train_test_split(X, y, test_size = 0.25)

clf = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.8500000000000001, n_estimators=100)),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=10, max_features=0.8, min_samples_leaf=17, min_samples_split=4, subsample=0.7500000000000001)
)

clf.fit(X_train, y_train)

accuracy = cross_val_score(clf, X, y, cv=5)
f1 = cross_val_score(clf, X, y, cv=5, scoring='f1')

print("Accuracy:", sum(accuracy)/len(accuracy))
print("f1:", sum(f1)/len(f1))

time_End = time.time()
print("Seconds to run:", time_End-time_Start )


# In[]
#==============================================================================
# Plot feature importance
#==============================================================================
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataframe and create labels
df = df_clean.copy()
target = 'OMXIPI'
tickers, df = petur.create_labels(target, df)

# Define features
features_to_use = df.columns.tolist()
print(features_to_use)
features_to_use.remove("Target")

y = df.Target
X = df[features_to_use]

# Ensemble classifier
clf = ExtraTreesClassifier(n_estimators=100, random_state = 0)
clf = clf.fit(df[features_to_use], df['Target'])

# Define features and importance
features = pd.DataFrame()
features['feature'] = df[features_to_use].columns
features['importance'] = clf.feature_importances_
features.sort_values(['importance'],ascending=False)

# Plot feature importance
sns.barplot(y = 'feature', x = 'importance', data=features.sort_values(by='importance', ascending=False))
plt.show()

