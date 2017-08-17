# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:25:11 2017

Write a program that prints the numbers from 1 to 100. 
But for multiples of three print "Fizz" instead of the number and 
for the multiples of five print "Buzz". 
For numbers which are multiples of both three and five print "FizzBuzz".



@author: petur
"""

# In[]
import petur_functions as petur
import pandas as pd


df_name = "Working_Dataset/Heatmap2.csv"
df = pd.read_csv(df_name, index_col='Date', header=0)

petur.corr_heatmap(df, save=True)