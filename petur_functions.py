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
    
def corr_heatmap(data, save=False):
    # https://pythonprogramming.net/stock-price-correlation-table-python-programming-for-finance/
    
    style.use('ggplot')
    
    df = pd.read_csv(data)
    df_corr = df.corr()
    if(save):
        name = data[:-4]
        name = name+'_corr.csv'
        df_corr.to_csv(name)
    
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
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    
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

def create_better_labels(stock, df, days):
    """
    Each model on a per company basis
    Input:
    - Ticker to predict
    - Entire dataframe
    - Days in the future to predict (default 7)
   """
   # list of ticker names
    stocks = df.columns.values.tolist()
    stocks.remove(stock)

    # percentage change days in the future
    df['Target_day_price'] = df[stock].shift(-days)
    df['Target_Change'] = (df[stock].shift(-days) - df[stock]) / df[stock]
    df = df.dropna()
            
    def f(x):
        if x['Target_Change'] > 0:
            return 1
        else:
            return 0

    df['Target'] = df.apply (lambda row: f(row),axis=1)
    
#    del df['Target_Change']
#    del df['Target_day_price']
    del df[stock]
    
    x = df[stocks]
    y = df['Target']
      
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