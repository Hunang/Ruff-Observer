# -*- coding: utf-8 -*-
# In[]
import numpy as np
import pandas as pd
import glob
import os
from functools import reduce
import xlrd
import csv
# In[]
#==============================================================================
# Load StockMarket Data (Kaupholl)
#==============================================================================
def load_market_Data():
    
    print("*** Loading Stock Market Data ***")
    path = 'Dataset_code/Kaupholl' # use your path
    allFiles = glob.glob(path + "/*.csv")
    
    df_final = pd.DataFrame()
    
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=False, header=0)
        df.DateTime = pd.to_datetime(df.DateTime, infer_datetime_format=True)
        df['Date'] = df['DateTime']
        del df['DateTime']
        df.set_index('Date', inplace = True)
        filename = os.path.basename(file_)[:-4]
        df.columns = [filename +'_Price', filename+'_Volume']
        
        # Add to final DF
        if df_final.empty:
            df_final = df
            #print("First file added to DF")
        else:
            #print("Joining to DF")
            df_final = df_final.join(df, how='outer')
        print(filename + ' added sucessfully')
    
    # Sort index and output file
    df = df_final.sort_index()
    df_final.to_csv(path+'_combined.csv', encoding='utf-8')
    print("******"+path+"\nFile saved sucessfully\n\n")
    return df_final

#df_test = load_market_Data()


# In[]
#==============================================================================
# Load Index Data
#==============================================================================
""" sample data
Date	         Price	      Open	      High	    Low	      Volume	  Change%
Jun 19, 2017	   12,849.50   12,845.25   12,893.50	 12,833.75	-	     0.76%
"""
def load_INDEX_Data():
    print("*** Loading Index Data ***")
    path = 'Dataset_code/INDEX' # use your path
    allFiles = glob.glob(path + "/*.xlsx")
    
    df_final = pd.DataFrame()
    
    for file_ in allFiles:
        # Read from file
        #print("\n",file_)
        df = pd.read_excel(file_, index_col=False, header=0)
        filename = os.path.basename(file_)[:-5]
        
        # Date to datetime and set as index
        df.Date =pd.to_datetime(df['Date'], format = '%b %d, %Y')
        #df.Date = pd.to_datetime(df.Date, infer_datetime_format=True)
        df.set_index('Date', inplace = True)
        
        #Rename columns
        df.columns = [filename +'_Price',filename +'_Open',
                      filename +'_High', filename +'_Low', filename+'_Volume', 'Change']   
        del df['Change']
        
        
        # Add to final DF
        if df_final.empty:
            df_final = df
            #print(filename+" First file added to DF")
        else:
            #print("Joining to DF")
            df_final = df_final.join(df, how='outer')
        print(filename + ' added sucessfully')
    
    # Sort index and output file
    df_final.sort_index()
    df_final.to_csv(path+'_combined.csv', encoding='utf-8')
    print("******"+path+"\nFile saved sucessfully\n\n")
    return df_final

#df_test = load_INDEX_Data()

# In[]
#==============================================================================
# Load FX Data
#==============================================================================
"""
Date	Price	Open	High	Low	Change%
Jun 19, 2017	76.75	76.85	77.03	76.59	-0.11%
"""
def load_FX_Data():
    print("*** Loading FX Data ***")
    path = 'Dataset_code/FX' # use your path
    allFiles = glob.glob(path + "/*.xlsx")
    
    df_final = pd.DataFrame()
    
    for file_ in allFiles:
        # Read from file
        #print("\n",file_)
        df = pd.read_excel(file_, index_col=False, header=0)
        filename = os.path.basename(file_)[:-5]
        
        # Date to datetime and set as index
        df.Date = pd.to_datetime(df['Date'], format = '%b %d, %Y')
        #df.Date = pd.to_datetime(df.Date, infer_datetime_format=True)
        df.set_index('Date', inplace = True)
        
        #Rename columns
        df.columns = [filename +'_Price',filename +'_Open',
                      filename +'_High', filename +'_Low', 'Change']   
        #Drop Change column
        del df['Change']
        
        # Add to final DF
        if df_final.empty:
            df_final = df
           # print(filename+" First file added to DF")
        else:
            #print("Joining to DF")
            df_final = df_final.join(df, how='outer')
        print(filename + ' added sucessfully')
    
    # Sort index and output file
    df_final.sort_index()
    df_final.to_csv(path+'_combined.csv', encoding='utf-8')
    print("******"+path+"\nFile saved sucessfully\n\n")
    return df_final

#df_test = load_FX_Data()
#load_FX_Data()

# In[]
def join_ALL_Data():
    print("*** Joining All Datasets ***")
    path = 'Dataset_code' # use your path
    allFiles = glob.glob(path + "/*.csv")
    
    df_final = pd.DataFrame()
    
    for file_ in allFiles:
        # Read from file
       # print("\n",file_)
        df = pd.read_csv(file_, index_col=False,header=0)
        df.Date = pd.to_datetime(df.Date, infer_datetime_format=True)
        df.set_index('Date', inplace = True)
        filename = os.path.basename(file_)[:-5]
            
        # Add to final DF
        if df_final.empty:
            df_final = df
           # print(filename+" First file added to DF")
        else:
            #print("Joining to DF")
            df_final = df_final.join(df, how='outer')
        print(filename + ' joined sucessfully to df_final')
    
    # Sort index and output file
    df_final.sort_index()
    df_final.to_csv(path+'_combined.csv', encoding='utf-8')
    print("***File saved")
    return df_final

#df_test = join_ALL_Data()

#join_ALL_Data()

# In[]

def load_all_data():
    print("\n\nLoading up datasets... \n")
    load_market_Data()
    load_FX_Data()
    load_INDEX_Data()
    print("All datasets loaded up sucessfully...")
    print("Joining all datasets...")
    join_ALL_Data()
    print("\nAll datasets joined sucessfully!")
    print("DONE!")
    
#load_all_data()
    
# In[]

def get_cleaned_data(file_, output_filename, date_start_, date_end_):
    #Load data
    df = pd.read_csv(file_, index_col=False,header=0)
    df.Date = pd.to_datetime(df.Date, infer_datetime_format=True)
    df = df[df['Date']>= date_start_]
    df = df[df['Date']< date_end_]
    df.set_index('Date', inplace = True)
    df.sort_index()
    
    df.to_csv(output_filename+'.csv', encoding='utf-8')
    
    return df

#df = get_cleaned_data("Dataset_code_combined.csv",'clean_data_FULL', "01/01/2010","01/01/2017")
    
# In[]

def load_market_price_Data():
    
    print("*** Loading Stock Market Data ***")
    path = 'Dataset_code/Kaupholl' # use your path
    allFiles = glob.glob(path + "/*.csv")
    
    df_final = pd.DataFrame()
    
    # Add all tickers
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=False, header=0)
        df.DateTime = pd.to_datetime(df.DateTime, infer_datetime_format=True)
        df['Date'] = df['DateTime']
        del df['DateTime']
        df.set_index('Date', inplace = True)
        filename = os.path.basename(file_)[:-4]
        df.columns = [filename, 'Volume']
        
        del df['Volume']
        
        # Add to final DF
        if df_final.empty:
            df_final = df
            #print("First file added to DF")
        else:
            #print("Joining to DF")
            df_final = df_final.join(df, how='outer')
        print(filename + ' added sucessfully')
        
    # Add OMXGI index
    index_file = 'Dataset_code/INDEX/OMXIPI.xlsx'
    df = pd.read_excel(index_file, index_col=False, header=0)
    filename = "OMXIPI_INDEX"
    df.Date = pd.to_datetime(df['Date'], format = '%b %d, %Y')
    df = df[['Date','Price']]
    df.columns = ['Date', 'OMXIPI_Index']
    #df.Date = pd.to_datetime(df.Date, infer_datetime_format=True)
    df.set_index('Date', inplace = True)
    
    df_final = df_final.join(df, how='outer')
        
    # Sort index and output file
    df_final = df_final.sort_index()
    df_final.to_csv(path+'_price.csv', encoding='utf-8')
    print("******\nFile saved sucessfully\n")
    
    
    return df_final

#load_market_price_Data()
#df = get_cleaned_data("Dataset_code/Kaupholl_price.csv",'Clean_price',"01/01/2016","01/01/2017")
    
    
# In[]
#==============================================================================
# Use this if you have some weird excel docs open
#==============================================================================
#os.remove('Dataset_code/INDEX\~$DAX.xlsx')