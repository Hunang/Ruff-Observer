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

""" TODO
Split training data where companies started trading
"""
# In[Libraries]
import pandas as pd
import numpy as np
import petur_functions as petur
import time
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

# Optimisation
from tpot import TPOTClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn import preprocessing

#Sklearn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#==============================================================================
# In[Load data]
#==============================================================================
# Load data from csv
name_df = "Working_Dataset/MAIN_FULL.csv"
df_raw = pd.read_csv(name_df, index_col='Date', header=0)

benchmark_name = "Scores/BENCHMARK.csv"
df_benchmark = pd.read_csv(benchmark_name, header=0)
first_column = df_benchmark.columns[0]
df_benchmark = df_benchmark.set_index(first_column, drop=True)
df_benchmark.dropna(axis=1, inplace=True)

importance_dic = pickle.load( open( "Pickles/importance_dic.p", "rb" ))

#==============================================================================
#In[Feature Engineering]
#==============================================================================
df_fe = df_raw.copy()

# Get list of tickers
tickers = petur.get_tickers()
tickers.append('OMXIPI')
#tickers = ['OMXIPI','MARL', 'OSSRu']   #45% of market cap + index

# Create MAs
sma_days = [5, 10, 20, 50]
fibbonaci = [1, 2, 3, 5, 8, 13, 21]
both =[1,2,3,5,8,10,13,20,21,30,50]

df_fe = petur.make_sma(df_fe, tickers, sma_days)
df_fe = petur.make_ewma(df_fe, tickers, sma_days)
df_fe = df_fe.dropna()  # Maybe do at end of labels

#==============================================================================
#In[Create labels]
#==============================================================================
df_tmp = df_fe.copy()

df_labels = pd.DataFrame()  #init new DF for labels
feature_list = pd.Series(df_tmp.columns.tolist()) #get list of features

days = [1, 3, 5, 10, 20]
name_list = []

for stock in tickers:
    for day in days:
        name = "%s-%s" %(stock, day)
        name_list.append(name)
        
        # Get labels
        x_train, y_train = petur.create_better_labels(stock, df_tmp, day, drop=False)
        df_labels[name] = y_train
    
#==============================================================================
#In[Test/Train split and normalise]
#==============================================================================
df_split = df_fe.copy()
x_train, x_test = petur.tr_ts_split(df_split)
y_train, y_test = petur.tr_ts_split(df_labels)

#Normalise data 
max_value_dic = {}
feature_list = pd.Series(x_train.columns.tolist())

# Get max value for all columns and store in dictionary
for feature in feature_list:
    max_value_dic[feature] = x_train[feature].max()

# Normalise training data
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train, columns = feature_list)
x_test = scaler.transform(x_test)
x_test = pd.DataFrame(x_test, columns = feature_list)


#==============================================================================
# In[Single classifier Prediciton]
#==============================================================================
time_Start =time.time()

#clf = SVC(kernel = 'rbf')
clf = AdaBoostClassifier()
clf = LinearSVC()
clf = SVC(kernel = 'rbf')
clf = LogisticRegression()
clf = KNeighborsClassifier()
clf = DecisionTreeClassifier()
clf = RandomForestClassifier()
clf = GaussianNB()
clf = SVC(kernel = 'poly') 
clf = SVC(kernel = 'sigmoid')
#clf = ExtraTreesClassifier()

scores = {}

for stock in tickers:
    acc = []
    for day in days:
        print("*" *100)
        name = "%s-%s" %(stock, day)
        features_to_use = importance_dic[name]
        print(name)
        
        #TrainingData
        x_tr = x_train[features_to_use].copy()    #With feature selection
        
        #TestingData
        x_testing = x_test[features_to_use].copy()      #With feature selection
        
        #Labels
        y_tr = y_train[name].copy()
        y_testing = y_test[name].copy()
        
        #Filter dataframe by dropping rows where stock wasn't on market
        to_drop = x_tr[stock].iloc[0]   #Gets the value of the first row
        x_tr = x_tr[x_tr[stock] != to_drop]     #Drops rows equal to the first row
        idx = x_tr.index.values.tolist()
        y_tr = y_tr.iloc[idx]
        
        print("* Dropped %i rows // %i remaining"  %  (len(x_train.index) - len(x_tr.index), len(x_tr.index)))

        # Train CLF and test
        clf.fit(x_tr, y_tr)
        accuracy = clf.score(x_testing, y_testing)
        
        print("* %f accuracy..." % accuracy)
        acc.append(accuracy)
    scores[stock] = acc

#Name of CLF used
clf_name = str(clf)[0:str(clf).find('(') ]

# Save to dataframe
scores = pd.DataFrame(scores).transpose()
scores.columns = days
scores.index.name = 'Tickers'

# Scores minus benchmark to get difference
df_difference = pd.DataFrame(scores.values-df_benchmark.values, columns=scores.columns)
df_difference.set_index(scores.index.values,inplace=True)

# Timing
time_End = time.time()
elapsed = time_End-time_Start
print("Total time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))

# Save to excel
excel_name = "Scores/%i_%s (%i sec).xlsx" %(time_Start, clf_name, elapsed)
writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')
scores.to_excel(writer, sheet_name='Scores')
df_benchmark.to_excel(writer, sheet_name='Benchmark')
df_difference.to_excel(writer, sheet_name='Improvements')

writer.save()

print(scores.mean())
print("Full average: ", scores.mean().mean())

# In[TPOT]
#==============================================================================
# Genetic Algorithm
#==============================================================================
time_Start =time.time()

#tickers = ['OMXIPI'] # For testing only

scores = {}
models = {}
models_full = {}

custom = petur.get_tpot_config()

tickers = ['OMXIPI']

for stock in tickers:
    acc = []
    mdl = []
    mdl_full = []
    for day in days:
        time_Start_loop = time.time()
        
        name = "%s-%s" %(stock, day)
        features_to_use = importance_dic[name]
        print("*" *100, "\n", name)

        #TrainingData
        x_tr = x_train[features_to_use].copy()    #With feature selection
        y_tr = y_train[name].copy()
        
        #TestingData
        x_testing = x_test[features_to_use].copy()      #With feature selection
        y_testing = y_test[name].copy()

        #Filter dataframe by dropping rows where stock wasn't on market
        to_drop = x_tr[stock].iloc[0]   #Gets the value of the first row
        x_tr = x_tr[x_tr[stock] != to_drop]     #Drops rows equal to the first row
        idx = x_tr.index.values.tolist()
        y_tr = y_tr.iloc[idx]
        
        print("* Dropped %i rows // %i remaining"  %  (len(x_train.index) - len(x_tr.index), len(x_tr.index)))
        
        #TPOT tuning
        tpot = TPOTClassifier(generations       =100, #iterations to the run pipeline
                              offspring_size    =100, #offspring to produce in each generation
                              population_size   =50, #individuals to retain in population every generation
                              verbosity         =2, 
                              n_jobs            =-1, 
                              max_time_mins     = 120, cv=5)
        # Test Accuracy
        tpot.fit(x_tr, y_tr)
        accuracy = tpot.score(x_testing, y_testing)
        acc.append(accuracy)
        
        # Save model parameters 
        pln_name = str(tpot._optimized_pipeline)    #Full name
        mdl_full.append(pln_name)   
        mdl.append(pln_name[0:pln_name.find('(')])  #Short name
               
        #Reset, time and report
        del tpot
        time_End_loop = time.time()
        elapsed = time_End_loop-time_Start_loop

        print("Final score:",accuracy)
        print("Time to run %s: %i sec / (%i min)" % (name, elapsed, (elapsed/60) ))
        print("*" *100)
        
    scores[stock] = acc
    models[stock] = mdl
    models_full[stock] = mdl_full

time_End = time.time()
elapsed = time_End-time_Start
    
# Reformat scores and models 
scores = pd.DataFrame(scores).transpose()
models = pd.DataFrame(models).transpose()
mdl_full = pd.DataFrame(mdl_full).transpose()

scores.columns = days
models.columns = days
mdl_full.columns = days

# Export scores and models to Excel
excel_name = "Scores/%i_TPOT_Score (%i sec).xlsx" %(time_Start, elapsed)
writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')
scores.to_excel(writer, sheet_name='Scores')
models.to_excel(writer, sheet_name='Models')
mdl_full.to_excel(writer, sheet_name='Models_full')
writer.save()

print("*" *100)
print("Saved as %s" % excel_name)
print("Total time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))

#==============================================================================
# In[Feature Importance, Adaboost]
"""
Done for now, shouldn't have to run this again
"""
#==============================================================================
time_Start =time.time()

importance_dic = {}

for stock in tickers:
    acc = []
    for day in days:
        name = "%s-%s" %(stock, day)
        print("*" * 100)
        print("Feature importance: %s" % name)
        
        # create labels to predict
        x_tr = x_train.copy()
        y_tr = y_train[name]
        
        #Filter dataframe by dropping rows where stock wasn't on market
        to_drop = x_tr[stock].iloc[0]   #Gets the value of the first row
        x_tr = x_tr[x_tr[stock] != to_drop]     #Drops rows equal to the first row
        idx = x_tr.index.values.tolist()
        y_tr = y_tr.iloc[idx]
        
        print("* Dropping training rows with 0 in %s column" % stock)
        print("* Dropped %i rows // %i remaining"  %  (len(x_train.index) - len(x_tr.index), len(x_tr.index)))
        
        # Define classifiers
        clf = AdaBoostClassifier()
        clf = clf.fit(x_tr, y_tr)
        
        # Define features and importance
        features = pd.DataFrame()
        features['feature'] = x_tr.columns
        features['importance'] = clf.feature_importances_
        features = features.replace(0,np.nan).dropna()
        features = features.sort_values(['importance'],ascending=False)
        features = features['feature']
        
        #Check if stock is is feature list. Add if not there
        if stock in features.values:
            print("Stock in there")
            importance_dic[name] = features
        else:
            print("Stock not there, appending")
            features = features.append(pd.Series(stock))
            importance_dic[name] = features
        print("* Found %i features to use" % len(features))
        
        # Plot feature importance
#        sns.barplot(y = 'feature', x = 'importance', data=features)
#        plt.show()
        
        # Log time and number of features
        time_loop_END=time.time()
        print("* Done...%i seconds elapsed\n" % (time_loop_END - time_Start))

# Timing
time_End = time.time()
elapsed = time_End-time_Start
print("Total time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))

"""
plt.figure()
plt.xlabel("Number of features selected")
plt.title("Number of features chosen with feature selection)")
plt.hist(n_featurs, bins = 20)
plt.show()
"""
piklName = "Pickles/importance_dic.p"
pickle.dump( importance_dic, open( piklName, "wb" ))

importance_dic = pickle.load( open( "Pickles/importance_dic.p", "rb" ))
print("Pickled dictionary at: %s" %piklName)

# In[Grid Search]

time_Start =time.time()
from sklearn.linear_model import LogisticRegressionCV

"""
# SVMs
clf = LinearSVC()
clf = SVC(kernel = 'rbf')
clf = SVC(kernel = 'poly') 
clf = SVC(kernel = 'sigmoid')

# Ensembles
clf = AdaBoostClassifier()
clf = DecisionTreeClassifier()
clf = RandomForestClassifier()

# Misc
clf = GaussianNB()
clf = LogisticRegression()
clf = KNeighborsClassifier()
"""

clf = LinearSVC()
clf = SVC(kernel = 'rbf')
clf = SVC(kernel = 'poly') 
clf = SVC(kernel = 'sigmoid')
param_grid_SVM = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
                  'gamma' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

clf = AdaBoostClassifier()
param_grid_ensemble = {'n_estimators': [10,30,50,70,100,150,200,300,400,500,750,1000],
                       'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.25, 1/3, 0.5, 0.75, 0.9, 1]}

# Logistic Regression
grid_search_MAXENT  = LogisticRegressionCV(Cs=20, cv=5, n_jobs=-1)

clf = KNeighborsClassifier(algorithm= 'brute', n_jobs =4)
param_grid_KNN = {'n_neighbors': [3,5,7,9,11,15,19,25,35,49,75,99],
                       'p': [1,2]}

clf = DecisionTreeClassifier()
param_grid_TREE = {'max_depth': [2,3,4,5,7,9,10,12,15,20,25,30,40,50], 
                  'min_samples_split': [2,3,4,5,6,7,8,9,10,12,15,20,25,30], 
                  'max_features': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0, 'auto', 'sqrt', 'log2'] 
                  }

clf = RandomForestClassifier()
param_grid_TREE = {'max_depth': [2, 5, 10, None],
              'max_features': [0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 'auto', 'sqrt', 'log2'], 
              'min_samples_split': [2,5,10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


grid_search_SVM  = GridSearchCV(clf, param_grid_SVM, cv=5, n_jobs =4)
grid_search_MAXENT  = LogisticRegressionCV(Cs=20, cv=5, n_jobs=-1)
grid_search_KNN = GridSearchCV(clf, param_grid_KNN, cv=5, n_jobs =4)
grid_search_TREE = GridSearchCV(clf, param_grid_TREE, cv=5, n_jobs =4)

scores = {}

for stock in tickers:
    acc = []
    for day in days:
        print("*" *100)
        name = "%s-%s" %(stock, day)
        features_to_use = importance_dic[name]
        print(name)
        
        #TrainingData
        x_tr = x_train[features_to_use].copy()    #With feature selection
        
        #TestingData
        x_testing = x_test[features_to_use].copy()      #With feature selection
        
        #Labels
        y_tr = y_train[name].copy()
        y_testing = y_test[name].copy()
        
        #Filter dataframe by dropping rows where stock wasn't on market
        to_drop = x_tr[stock].iloc[0]   #Gets the value of the first row
        x_tr = x_tr[x_tr[stock] != to_drop]     #Drops rows equal to the first row
        idx = x_tr.index.values.tolist()
        y_tr = y_tr.iloc[idx]
        
        print("* Dropped %i rows // %i remaining"  %  (len(x_train.index) - len(x_tr.index), len(x_tr.index)))

        # Train CLF and test
        grid_search_TREE.fit(x_tr, y_tr)
        accuracy = grid_search_TREE.score(x_testing, y_testing)
        
        print("* %f accuracy..." % accuracy)
        print("Best parameters:\n", grid_search_TREE.best_params_)
        acc.append(accuracy)
    scores[stock] = acc

#Name of CLF used
clf_name = str(clf)[0:str(clf).find('(') ]

# Save to dataframe
scores = pd.DataFrame(scores).transpose()
scores.columns = days
scores.index.name = 'Tickers'

# Scores minus benchmark to get difference
df_difference = pd.DataFrame(scores.values-df_benchmark.values, columns=scores.columns)
df_difference.set_index(scores.index.values,inplace=True)

# Timing
time_End = time.time()
elapsed = time_End-time_Start
print("Total time to run: %i sec / (%i min)" % (elapsed, (elapsed/60) ))

# Save to excel
excel_name = "Scores/%i_%s (%i sec).xlsx" %(time_Start, clf_name, elapsed)
writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')
scores.to_excel(writer, sheet_name='Scores')
df_benchmark.to_excel(writer, sheet_name='Benchmark')
df_difference.to_excel(writer, sheet_name='Improvements')

writer.save()

print("*"*100)
print(clf)
print(scores.mean())
print("Full average: ", scores.mean().mean())
