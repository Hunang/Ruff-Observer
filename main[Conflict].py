# -*- coding: utf-8 -*-
#==============================================================================
# In[Imports]
#==============================================================================
import petur_functions as petur
#import data_manipulation as load

import pandas as pd
import numpy as np
import time
import operator
from collections import Counter
from datetime import datetime

from tpot import TPOTClassifier
from tpot.builtins import StackingEstimator

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn import cross_validation as cv
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.preprocessing import FunctionTransformer
from copy import copy


#==============================================================================
# In[Load Data]
#==============================================================================

df_train = pd.read_csv("MAIN_TRAIN_2.CSV", index_col='Date', header=0)


# ## NOTE:
#==============================================================================
# df_test is a held-out test set!!!
# SHOULD NOT BE USED, EXCEPT FOR TESTING AT KEY POINTS IN PROJECT
#==============================================================================

df_test = pd.read_csv("MAIN_TEST_2.CSV", index_col='Date', header=0)

#==============================================================================

#==============================================================================
# In[Do ML]
#==============================================================================

# Machine Learning - One off
#==============================================================================
time_Start =time.time()

df = df_train.copy()

target = 'OMXIPI'

# Define features to use
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

accuracy = cross_val_score(clf, X, y, cv=10) #,average='binary')
f1 = cross_val_score(clf, X, y, cv=10, scoring='f1') #, average='binary')

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

# In[Genetic algorithm - Classification]
# Generate Genetic algorithm output == TEST DATA 
#==============================================================================

time_Start = time.time()

train_data = df_train.copy()
test_data = df_test.copy()

features = petur.get_tickers()
features.append('OMXIPI')

# Train test split 
y_train = train_data.Target
X_train = train_data[features]

y_test = test_data.Target
X_test = test_data[features]

tpot = TPOTClassifier(generations=50, population_size=50, 
                      verbosity=2, scoring='f1', 
                      n_jobs = -1, cv=5)

tpot.fit(X_train, y_train)
print("Final score:",tpot.score(X_test, y_test))

name = "TPOT Classification - "+datetime.now().strftime('%m-%d_%H-%M')
name = name+".py"
tpot.export(name)

time_End = time.time()
print("Seconds to run:", time_End-time_Start )

# In[Implement GA - Classification]
#==============================================================================

time_Start =time.time()
# Load dataframe and create labels

train_data = df_train.copy()
#test_data = df_test.copy()

features = petur.get_tickers()
features.append('OMXIPI')

# Train test split 
y_train = train_data.Target
X_train = train_data[features]

y_test = test_data.Target
X_test = test_data[features]

exported_pipeline = make_pipeline(
    SelectFromModel(estimator=ExtraTreesClassifier(criterion="entropy", max_features=1.0), 
                    threshold=0.05),
    GradientBoostingClassifier(learning_rate=1.0, max_depth=7, 
                               max_features=0.6, min_samples_leaf=20, 
                               min_samples_split=8, subsample=0.25))
exported_pipeline.fit(X_train,y_train)
predictions = exported_pipeline.predict(X_test)

print('True spread:     ', Counter(y_test))
print('Predicted spread:', Counter(predictions))
petur.print_evaluation(y_test,predictions)
df_predict = pd.DataFrame({'Predictions': predictions, 'True_label': y_test})

time_End = time.time()
print("Seconds to run:", time_End-time_Start )

#==============================================================================
# In[Genetic algorithm - Regression]
#==============================================================================
from tpot import TPOTRegressor

time_Start = time.time()

train_data = df_train.copy()
test_data = df_test.copy()

features = petur.get_tickers()
features.append('OMXIPI')

# Train test split 
y_train = train_data.Target
y_train2 = train_data.OMXIPI.shift(-7)
y_train2 = y_train2[:-7]
X_train = train_data[features]
X_train2 = X_train[:-7]

y_test = test_data.Target
y_test2 = y_test[:-7]
X_test = test_data[features]
X_test2 = X_test[:-7]


tpot = TPOTRegressor(generations=50, population_size=60, 
                     verbosity=2, n_jobs = -1, cv=5, 
                     max_time_mins = 60)
tpot.fit(X_train2, y_train2)
print(tpot.score(X_test2, y_test2))

print("Final score:",tpot.score(X_test, y_test))

name = "TPOT Regression - "+datetime.now().strftime('%m-%d_%H-%M')
name = name+".py"
tpot.export(name)

time_End = time.time()
print("Seconds to run:", time_End-time_Start )
#==============================================================================
# In[Plot feature importance]
#==============================================================================
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataframe and create labels
df = df_train.copy()
target = 'OMXIPI'
tickers, df = petur.create_labels(target, df)

# Define features
tickers = petur.get_tickers()
features_to_use = tickers
print(features_to_use)
#features_to_use.remove("Target")

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
