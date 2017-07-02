import numpy as np
import pandas as pd
import time
from scipy.stats import itemfreq

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


def print_evaluation(true_label, prediction, clf_type):
    
    if(clf_type == "Classification"):
        accuracy = accuracy_score(true_label, prediction)
        metrics = precision_recall_fscore_support(true_label, prediction, average='binary')
    
        print("Accuracy:  ", accuracy)
        print("Precision: ", metrics[0])
        print("Recall:    ", metrics[1])
        print("F1-score:  ", metrics[2])
    elif(clf_type == "SVR"): 
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