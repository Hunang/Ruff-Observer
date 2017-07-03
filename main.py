# -*- coding: utf-8 -*-
#from join_all_data import load_all_data
import petur_functions as petur
import join_all_data as p_join

#import sentdex_preprocessing as sentdex

#load_all_data()

# In[]
#==============================================================================
# Load data
#==============================================================================
p_join.load_market_price_Data()
p_join.get_cleaned_data("Dataset_code/Kaupholl_price.csv",'Clean_price',"01/01/2016","01/01/2017")
x=1
