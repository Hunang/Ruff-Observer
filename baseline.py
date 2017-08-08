# -*- coding: utf-8 -*-
# In[]
#==============================================================================
# Libraries
#==============================================================================
# Custom libraries
import petur_functions as petur

# Base libraries
import pandas as pd

# Machine learning libraries
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC

# In[Load data]
#==============================================================================
# Load data
#==============================================================================
df_train = pd.read_csv("MAIN_TRAIN.CSV", index_col='Date', header=0)
df_test = pd.read_csv("MAIN_TEST.CSV", index_col='Date', header=0)

# In[Timeseries cv]
#==============================================================================
# Timeseries cv
#==============================================================================
# Load data and define features
df = df_train.copy()
features = petur.get_tickers()
features.append('OMXIPI')

# Train test split 
y = df.Target
X = df[features]

# CLF 
clf = SVC()

# CV
ac,pr,re,f1 = petur.timeseries_cv(X,y,clf, splits=5)#, verbosity=False)

"""
Final baseline outcome on 09/July/2016
- CV of 10 folds 
- SVC(kernel='rbf') and 
- Ticker price of all companies to 
- Predict index price changes by 2% 7 days in the future

Accuracy:   0.6728
Precision:  0.381768929984
Recall:     0.495955797689
F1-score:   0.412887585056
"""
# In[Timeseries test set]
#==============================================================================
# Timeseries test set
#==============================================================================
train_data = df_train.copy()
test_data = df_test.copy()

features = petur.get_tickers()
features.append('OMXIPI')

# Train test split 
y_train = train_data.Target
X_train = train_data[features]

y_test = test_data.Target
X_test = test_data[features]

# CLF 
clf = svm.LinearSVC()
clf = SVC()

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

petur.print_evaluation(y_test, prediction)

df_predict = pd.DataFrame({'Predictions': prediction, 'True_label': y_test})
"""
Final baseline outcome on 09/July/2016
- CV of 10 folds 
- SVC(kernel='rbf') and 
- Ticker price of all companies to 
- Predict index price changes by 2% 7 days in the future

Accuracy:   0.773638968481
Precision:  0.696428571429
Recall:     0.386138613861
F1-score:   0.496815286624
"""


# In[]
""" OLDER code
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

   
"""
