#==============================================================================
# Classifier Tests
#==============================================================================

# In[2]:
    
import petur_functions as petur

import numpy as np
import pandas as pd
import datetime
import time
from scipy.stats import itemfreq

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


#==============================================================================
# Load data
#==============================================================================

# In[3]:

#Define testfile
test_file = 'Dataset/INDEX/ICEX Main (OMXIPI).xlsx'

# Create dataframe
df = pd.read_excel(test_file)

#==============================================================================
# PreProcessing
#==============================================================================

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
df['EWMA'] = df.Price_shift.ewm(com=0.5,min_periods=0,adjust=True,ignore_na=False).mean()

# Drop NAN
# TODO: Find better approach
df = df.dropna()


#==============================================================================
# ML - PreProcessing
#==============================================================================

# In[5]:

df_new = df.copy()

# Create labels [0: "Down", 1: "Up"]
le = preprocessing.LabelEncoder()
df_new['UpDown'] = np.where(df_new['Return']>=0, 'Up', 'Down')
df_new.UpDown = le.fit(df_new.UpDown).transform(df_new.UpDown)


# Define training features and label
features = df_new.columns[10:-1]
X = preprocessing.scale(df_new[features])    
y = df_new.UpDown   

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print(len(X), "Total datapoints")
print(len(X_train), "Training datapoints")
print(len(X_test), " Testing datapoints")
print(len(df_new[features].columns), "Features")
print(itemfreq(df_new.UpDown))


#==============================================================================
# Ensemble classifier
#==============================================================================

# In[7]:

time_Start =time.time()

# Classifier
clf = ExtraTreesClassifier(n_estimators=1000)
clf = clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)
petur.print_class_evaluation(y_test, prediction)

time_End = time.time()
print("Seconds to run:", time_End-time_Start )


#==============================================================================
# Linear SVM Classifier
#==============================================================================

# In[8]:

# Start timer
time_Start =time.time()

# Classifier
clf = LinearSVC()
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)
petur.print_class_evaluation(y_test, prediction)

# End timer and results
time_End = time.time()
print("Seconds to run:", time_End-time_Start )


# In[8]:

print(itemfreq(prediction))
print(prediction)


# In[9]:

# Start timer
time_Start =time.time()

# Classifier
clf = SVC()
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)
petur.print_class_evaluation(y_test, prediction)

# End timer and results
time_End = time.time()
print("Seconds to run:", time_End-time_Start )


# In[10]:

print(itemfreq(prediction))
print(prediction)


#==============================================================================
# Feature importance
#==============================================================================

# In[11]:

# Define features and importance
df_feature = pd.DataFrame()
df_feature['Features'] = df_new[features].columns
weights = clf.coef_ * clf.coef_
weights = abs(clf.coef_)
df_feature['Importance'] = weights[0]
df_feature.sort(['Importance'],ascending=False)
top5 = df_feature.Features[0:5]

# Plot feature importance
sns.barplot(y = 'Features', x = 'Importance', data=df_feature.sort_values(by='Importance', ascending=False))
plt.title("Linear SVM feature importance", fontsize=18)
plt.show()


# In[15]:

weights = clf.coef_ * clf.coef_
print(weights)
print(df_feature['Features'])


# In[16]:

df.head()


# In[36]:

df.Price
df.Date
ts = pd.Series(df.Price.values, index = df.Date)


# In[37]:

ts.plot()
plt.show()


# In[ ]:



