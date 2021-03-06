import numpy as np
import pandas as pd
import time
from scipy.stats import itemfreq

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from collections import Counter

from sklearn import preprocessing 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def print_evaluation(true_label, prediction, clf_type="Classification"):
    
    if(clf_type == "Classification"):
        metrics = precision_recall_fscore_support(true_label, prediction, average='binary')
    
        print("Accuracy:  ", accuracy_score(true_label, prediction))
        print("Precision: ", metrics[0])
        print("Recall:    ", metrics[1])
        print("F1-score:  ", metrics[2])
    elif(clf_type == "Regression"): 
        metrics = mean_squared_error(true_label, prediction)
        print(metrics)
    
def index_preprocess(df):
    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['Date']).apply(lambda x: x.date())
    df = df.sort('Date', ascending=True)
    
    # Create return
    df['Return']        = df['Price'].pct_change(periods=1)
    
    # Create shifted price and return (% returns from yesterday)
    """
                Date   Price    Return  Return_shift
    1806  2010-03-16  549.73 -0.004509      0.010504
    1805  2010-03-17  556.05  0.011497     -0.004509
    1804  2010-03-18  564.50  0.015196      0.011497
    """
    df['Return_shift']  = df.Return.shift(1)
    df['Price_shift']   = df.Price.shift(1)
    
    # Returns are based on the last day's return and the time
    # So Return3 for t0 is %change from t-4 to t-1
    # Same applies for averages
    
    df['Return3']       = df['Price_shift'].pct_change(periods=3)
    df['Return5']       = df['Price_shift'].pct_change(periods=5)
    df['Return10']      = df['Price_shift'].pct_change(periods=10)
    df['Return20']      = df['Price_shift'].pct_change(periods=20)
    df['Return50']      = df['Price_shift'].pct_change(periods=50)
    
    # Create moving price averages
    df['3DaySMA']       = df['Price_shift'].rolling(center=False,window =3).mean()
    df['5DaySMA']       = df['Price_shift'].rolling(center=False,window =5).mean()
    df['10DaySMA']      = df['Price_shift'].rolling(center=False,window=10).mean()
    df['20DaySMA']      = df['Price_shift'].rolling(center=False,window=20).mean()
    df['50DaySMA']      = df['Price_shift'].rolling(center=False,window=50).mean()
    
    # Exponential Weighted Moving Average
    df['EWMA'] = df.Price_shift.ewm(com=0.5,min_periods=0,adjust=True,ignore_na=False).mean()
    
    # Drop NAN
    # TODO: Find better approach
    df = df.dropna()
    return df

def print_ts_vs_prediction(real_price, predicted_price):
    # Plot TS
    plt.plot(real_price.get_values())
    plt.plot(predicted_price.get_values())
    plt.legend(['Price', 'Prediction'])
    plt.title('Time series', fontsize=15)
    plt.show()
    
def corr_heatmap(df, save=False):
    # https://pythonprogramming.net/stock-price-correlation-table-python-programming-for-finance/
    
    style.use('ggplot')
    
    df_corr = df.corr()
    if(save):
        df_corr.to_csv("Correlation_Heatmap_values.csv")
    
    data1 = df_corr.values
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    
    fig1.colorbar(heatmap1)
    
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels, size='x-large')
    ax1.set_yticklabels(row_labels, size='x-large')
    
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()

    
def get_tickers():
    tickers = ['EIK','EIM','GRND','HAGA','ICEAIR','MARL',
               'N1','NYHR','OSSRu','REGINN','REITIR','SIMINN',
               'SJOVA','TM','VIS','VOICE']
    return tickers

# In[]
def create_labels(ticker, df, days=7, change=0.02, binary=False):
    """
    Each model on a per company basis
    Input:
    - Ticker to predict
    - Entire dataframe
    - Days in the future to predict (default 7)
   """
    
   # list of ticker names
    tickers = df.columns.values.tolist() 

    # percentage change days in the future
    df['Target_day_price'] = df[ticker].shift(-days)
    df['Target_Change'] = (df[ticker].shift(-days) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
            
    def f(x):
        if x['Target_Change'] < change *-1:
            return -1
        elif x['Target_Change'] > change:
            return 1
        else:
            return 0
    def b(x):
        if x['Target_Change'] < change *-1:
            return 1
        elif x['Target_Change'] > change:
            return 1
        else:
            return 0

    if(binary):
        df['Target'] = df.apply (lambda row: b(row),axis=1)
    else:
        df['Target'] = df.apply (lambda row: f(row),axis=1)
    
    del df['Target_Change']
    del df['Target_day_price']
      
    return tickers, df

def create_better_labels(stock, df, days, drop=True):
    """
    Each model on a per company basis
    Input:
    - Ticker to predict
    - Entire dataframe
    - Days in the future to predict (default 7)
   """
   # list of ticker names
    stocks = df.columns.values.tolist()

    # percentage change days in the future
    df['Target_day_price'] = df[stock].shift(-days)
    df['Target_Change'] = (df[stock].shift(-days) - df[stock]) / df[stock]
    
    if drop:
        df = df.dropna()

#    df['Target'] = df['Target_Change'].apply(lambda row: f(row))
    stuff = df['Target_Change'].apply(lambda x: 1 if x>0 else 0)
#    stuff = stuff.copy()
#    df.loc[:,'Target'] = stuff
    
    x = df[stocks]
    y = stuff
      
    return x, y
    

#df = pd.read_csv('df_clean.csv', index_col=False, header=0)
#x,df = create_labels('OMXIPI', df, days=7, change=0.02)

# In[]
#==============================================================================
# Timeseries CV
#==============================================================================
def timeseries_cv(X, y, clf, splits=10, verbosity=True):
    # Init scores arrays
    accuracy    = []
    precision   = []
    recall      = []
    f1_score    = []
    
    # Make splits and loop over them
    tscv = TimeSeriesSplit(n_splits=splits)
    for train_index, test_index in tscv.split(X):
        clf_loop = clf
        # Define start and stop indexes for timeseries crossvalidation
        train_start = train_index.min()
        train_stop = train_index.max()
        test_start = train_stop+1
        test_stop = test_index.max()
        
        # Split data based on start and stop
        X_train, X_test = X[train_start:train_stop], X[test_start:test_stop]
        y_train, y_test = y[train_start:train_stop], y[test_start:test_stop]
        
        # Make prediction
        clf_loop.fit(X_train, y_train)
        predictions = clf_loop.predict(X_test)
        
        # Append results to arrays
        metrics = precision_recall_fscore_support(y_test, predictions)#, average='binary')
        accuracy.append(accuracy_score(y_test, predictions))
        precision.append(np.mean(metrics[0]))
        recall.append(np.mean(metrics[1]))
        f1_score.append(np.mean(metrics[2]))
        
        # Results
        if verbosity:
            print("Training: %s - %s" %(train_start,train_stop))
            print("Testing: %s - %s" %(test_start,test_stop))
            print_evaluation(y_test, predictions)
            print('True spread:     ', Counter(y_test))
            print('Predicted spread:', Counter(predictions))
            print("******************************************\n")
    
    # Print summary stats
    print("\n******************************************")
    print("******************************************")
    print("Final outcome from CV of %s folds" %splits)
    print("Accuracy:  ", np.mean(accuracy))
    print("Precision: ", np.mean(precision))
    print("Recall:    ", np.mean(recall))
    print("F1-score:  ", np.mean(f1_score))

    # Return 
    return accuracy, precision, recall, f1_score

# In[Moving Averages]
def make_sma(passed_df, feature_list, day_list, clean = False ):
    
    for feature in feature_list:
        for day in day_list:
            feature_name = "%s_%iSMA" %(feature, day)
            passed_df[feature_name] = passed_df[feature].rolling(center=False,window =day).mean()
    if(clean):
        return passed_df.dropna()
    else: 
        return passed_df

def make_ewma(passed_df, feature_list, day_list, clean = False):
    for feature in feature_list:
        for day in day_list:
            feature_name = "%s_%iEWMA" %(feature, day)
            passed_df[feature_name] = passed_df[feature].ewm(com=0.5,min_periods=0,adjust=True,ignore_na=False).mean()
    if(clean):
        return passed_df.dropna()
    else: 
        return passed_df
    
# In[Train/Test split of df]
    
def tr_ts_split(df, date="01/01/2017"):
    #def new dfs
    df_train = df.copy()
    df_test = df.copy()
    
    #Filter training
    df_train['Date']=df_train.index
    df_train.Date = pd.to_datetime(df_train.Date, infer_datetime_format=True)
    df_train =df_train[df_train['Date'] < date]
    del df_train['Date']
    
    
    #Filter test
    df_test['Date']=df_test.index
    df_test.Date = pd.to_datetime(df_test.Date, infer_datetime_format=True)
    df_test =df_test[df_test['Date'] >= date]
    del df_test['Date']    
    
    return df_train, df_test   

def get_tpot_config():
    tpot_config = {
        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf':  range(1, 21),
            'bootstrap': [True, False]
        },     
        'sklearn.ensemble.GradientBoostingClassifier': {
            'n_estimators': [100],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'subsample': np.arange(0.05, 1.01, 0.05),
            'max_features': np.arange(0.05, 1.01, 0.05)
        }, 
        'sklearn.svm.LinearSVC': {
            'penalty': ["l1", "l2"],
            'loss': ["hinge", "squared_hinge"],
            'dual': [True, False],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
        },     
        'sklearn.linear_model.LogisticRegression': {
            'penalty': ["l1", "l2"],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False]
        },
        'xgboost.XGBClassifier': {
            'n_estimators': [100],
            'max_depth': range(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': np.arange(0.05, 1.01, 0.05),
            'min_child_weight': range(1, 21),
            'nthread': [1]
        },
        'sklearn.svm.SVC': {
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.], 
            'kernel': ['poly', 'rbf', 'sigmoid'], 
            'degree': range(1,5)
        },
        
        'sklearn.decomposition.PCA': {
            'svd_solver': ['randomized'],
            'iterated_power': range(1, 11)
        },
        'sklearn.decomposition.FastICA': {
            'tol': np.arange(0.0, 1.01, 0.05)
        },
        'sklearn.kernel_approximation.Nystroem': {
            'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
            'gamma': np.arange(0.0, 1.01, 0.05),
            'n_components': range(1, 11)
        },
        'sklearn.kernel_approximation.RBFSampler': {
            'gamma': np.arange(0.0, 1.01, 0.05)
        },
        # Selectors
        'sklearn.feature_selection.SelectFwe': {
            'alpha': np.arange(0, 0.05, 0.001),
            'score_func': {
                'sklearn.feature_selection.f_classif': None
            }
        },
    
        'sklearn.feature_selection.SelectPercentile': {
            'percentile': range(1, 100),
            'score_func': {
                'sklearn.feature_selection.f_classif': None
            }
        },
    
        'sklearn.feature_selection.VarianceThreshold': {
            'threshold': np.arange(0.05, 1.01, 0.05)
        },
    
        'sklearn.feature_selection.RFE': {
            'step': np.arange(0.05, 1.01, 0.05),
            'estimator': {
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    'max_features': np.arange(0.05, 1.01, 0.05)
                }
            }
        },
    
        'sklearn.feature_selection.SelectFromModel': {
            'threshold': np.arange(0, 1.01, 0.05),
            'estimator': {
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    'max_features': np.arange(0.05, 1.01, 0.05)
                }}}}
    return tpot_config