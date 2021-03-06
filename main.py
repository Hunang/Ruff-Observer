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
# In[Data Exploration]

df_train.head()
df_train['OMXIPI'].plot(title="OMXIPI Icelandic Stock Market Index")

# In[Machine Learning - One off]
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
elapsed = time_End-time_Start
print("Time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))

# In[Classification - Genetic algorithm Generation]
#==============================================================================
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

time_End = time.time()
elapsed = time_End-time_Start
print("Time to run: %i sec / (%f min)" % (elapsed, (elapsed/60) ))

name = "TPOT/TPOT Classification - "+datetime.now().strftime('%m-%d_%H-%M')
name = name+"-"+elapsed+".py"
tpot.export(name)

# In[Classification - Genetic algorithm Testing]
#==============================================================================
# Implement Genetic Algorithm
#==============================================================================
time_Start =time.time()
# Load dataframe and create labels

train_data = df_train.copy()
test_data = df_test.copy()

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
elapsed = time_End-time_Start
print("Time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))

#==============================================================================
# In[Regression - Genetic algorithm Generation]

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


tpot = TPOTRegressor(generations=100, population_size=100, 
                     verbosity=2, n_jobs = -1, cv=5, 
                     max_time_mins = 100)
tpot.fit(X_train2, y_train2)
print(tpot.score(X_test2, y_test2))

print("Final score:",tpot.score(X_test, y_test))

time_End = time.time()
elapsed = time_End-time_Start
print("Time to run: %i sec / (%f min)" % (elapsed, (elapsed/60) ))

name = "TPOT/TPOT Regression - "+datetime.now().strftime('%m-%d_%H-%M')
name = name+"-"+elapsed+".py"
tpot.export(name)
#==============================================================================
#==============================================================================
# In[Regression - Genetic algorithm Implementation]
from sklearn.linear_model import LassoLarsCV
from sklearn.svm import LinearSVR
#==============================================================================
# Implement Genetic Algorithm Regression
#==============================================================================
time_Start =time.time()
# Load dataframe and create labels

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

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    LinearSVR(C=0.0001, dual=False, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.0001)
)

exported_pipeline.fit(X_train2,y_train2)
predictions = exported_pipeline.predict(X_test2)

#
#print('True spread:     ', Counter(y_test2))
#print('Predicted spread:', Counter(predictions))
#petur.print_evaluation(y_test2,predictions)
df_predict = pd.DataFrame({'Predictions': predictions, 'True_label': y_test2})

time_End = time.time()
elapsed = time_End-time_Start
print("Time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))

#==============================================================================
# In[Plot feature importance]
#==============================================================================
# Plot feature importance
#==============================================================================
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


# Load dataframe and create labels
df = df_train.copy()
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

# In[==TEST BED==]
time_End = time.time()
elapsed = time_End-time_Start
print("Time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))