
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import datetime
import time

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


# In[5]:

#Define testfile
test_file = 'Dataset/INDEX/ICEX Main (OMXIPI).xlsx'

# Create dataframe

df = pd.read_excel(test_file)
df_test = df.copy()

df.head()


# # Pre processing

# In[3]:

"""
# Below code is not in use at the moment
# Convert '6.20M' to int millions
def to_millions(data):
    if data[-1]== 'M':
        data=float(data[:-1])
        return int(data * 1000000)
    else:
        return int(data * 1000000)
    
# Remove rows with missing values for Volume 
# Might not need, volume might not be a good indicator as it reduces the number of datapoints from 1800 > 600 ish
df =df[df_test.Volume != '-']
# Convert volume from '6.20M' to '6,200,000'
df.Volume = df.Volume.apply(lambda x: to_millions(x))
"""


# In[4]:

# Convert timestamp to datetime
df['Date'] = pd.to_datetime(df['Date']).apply(lambda x: x.date())
df = df.sort('Date', ascending=True)

# Create returns
df['Return'] = df['Price'].pct_change(periods=1)
df['Return_shift'] = df.Return.shift(1)
df['Price_shift'] = df.Price.shift(1)
df['Return3'] = df['Price_shift'].pct_change(periods=3)
df['Return5'] = df['Price_shift'].pct_change(periods=5)
df['Return10'] = df['Price_shift'].pct_change(periods=10)
df['Return20'] = df['Price_shift'].pct_change(periods=20)
df['Return50'] = df['Price_shift'].pct_change(periods=50)

# Create moving price averages
df['3DaySMA'] = df['Price_shift'].rolling(center=False,window =3).mean()
df['5DaySMA'] = df['Price_shift'].rolling(center=False,window =5).mean()
df['10DaySMA'] = df['Price_shift'].rolling(center=False,window=10).mean()
df['20DaySMA'] = df['Price_shift'].rolling(center=False,window=20).mean()
df['50DaySMA'] = df['Price_shift'].rolling(center=False,window=50).mean()

# Exponential Weighted Moving Average
df['EWMA'] = pd.ewma(df.Price_shift, com=0.5)

# Drop NAN
# TODO: Find better approach
df = df.dropna()

# Show
df.head()


# In[6]:

# Check for na values
#df.isnull().sum()


# # Select features to use in ML

# In[7]:

from sklearn import preprocessing 

df_new = df.copy()

# Create labels [0: "Down", 1: "Up"]
le = preprocessing.LabelEncoder()
df_new['UpDown'] = np.where(df_new['Return']>=0, 'Up', 'Down')
df_new.UpDown = le.fit(df_new.UpDown).transform(df_new.UpDown)

#print(df_new.columns)


# In[8]:

# Define training features and label
features = df_new.columns[10:-1]
X = df_new[features]    
y = df_new.UpDown    

# Df is 1786 items
n_split = 1500

# Define training and testing
X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]


# In[9]:

print(len(X_train), "training datapoints")
print(len(X_test), " testing datapoints")
print(len(X.columns))


# # Random forest model

# In[25]:

def print_evaluation(true_label, prediction):
    accuracy = accuracy_score(y_test, prediction)
    other = precision_recall_fscore_support(y_test, prediction, average='binary')

    print("Accuracy:  ", accuracy)
    print("Precision: ", other[0])
    print("Recall:    ", other[1])
    print("F1-score:  ", other[2])


# In[26]:

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Ensemble classifier
clf = ExtraTreesClassifier(n_estimators=100, random_state = 0)
clf = clf.fit(X_train, y_train)


# In[ ]:




# In[27]:

# Prediction
prediction = clf.predict(X_test)
print_evaluation(y_test, prediction)


# In[29]:

clf_cv = ExtraTreesClassifier(n_estimators=100, random_state = 0)
clf_cv = clf_cv.fit(X, y)
scores = cross_val_score(clf_cv, X, y, cv=10)
print(scores)
print(scores.mean())


# # Check for feature importance

# In[157]:

# Ensemble classifier
clf_imp = ExtraTreesClassifier(n_estimators=100, random_state = 0)
clf_imp = clf_imp.fit(X, y)

# Define features and importance
features = pd.DataFrame()
features['feature'] = X.columns
features['importance'] = clf_imp.feature_importances_
features.sort(['importance'],ascending=False)

# Plot feature importance
sns.barplot(y = 'feature', x = 'importance', data=features.sort_values(by='importance', ascending=False))
plt.show()


# In[141]:

# Code from Siraj
# Not working
def predict_prices(dates, prices, x):
	dates = np.reshape(dates,len(dates),1)

	svr_lin  = SVR(kernel = 'linear', C=1e3)
	svr_poly = SVR(kernel = 'poly',   C=1e3, degree = 2)
	svr_rbf  = SVR(kernel = 'rbf',    C=1e3, gamma  = 0.1)
	svr_lin.fit(dates,prices)
	svr_poly.fit(dates,prices)
	svr_rbf.fit(dates,prices)

	plt.scatter(dates, prices, color='black', label='Data')
	plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
	plt.plot(dates, svr_lin.predict(dates, color='green', label='Linear Model'))
	plt.plot(dates, svr_poly.predict(dates, color='blue', label='Polynomial Model'))
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('SVM')
	plt.legend()
	plt.show	

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


# # Support Vector Machine

# In[30]:

from sklearn.svm import SVC

def performSVMClass(X_train, y_train, X_test, y_test):
    """
    SVM binary Classification
    """
    clf = SVC()
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy


# In[31]:

accuracy = performSVMClass(X_train, y_train, X_test, y_test)
print(accuracy)


# In[ ]:

time_Start =time.time()

clf_SVC = SVC(kernel='poly')
#clf_SVC = clf_SVC.fit(X, y)
scores = cross_val_score(clf_SVC, X, y, cv=5)
print(scores)
print(scores.mean())

time_End = time.time()

print("Seconds to run:", time_End-time_Start )


# # Polynomial SVM

# In[6]:

""" This is stuck in training. Something wrong, have to see what
# Start timer
time_Start =time.time()

# Classifier
clf = SVC(kernel='poly')
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)
print_evaluation(y_test, prediction)

# End timer and results
time_End = time.time()
print("Seconds to run:", time_End-time_Start )
"""


# # RBF SVM

# In[ ]:

# Start timer
time_Start =time.time()

# Classifier
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)
print_evaluation(y_test, prediction)

# End timer and results
time_End = time.time()
print("Seconds to run:", time_End-time_Start )


# # Sigmoid SVM

# In[ ]:

# Start timer
time_Start =time.time()

# Classifier
clf = SVC(kernel='sigmoid')
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)
print_evaluation(y_test, prediction)

# End timer and results
time_End = time.time()
print("Seconds to run:", time_End-time_Start )

