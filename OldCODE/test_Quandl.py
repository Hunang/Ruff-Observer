# -*- coding: utf-8 -*-
# In[]
import quandl

quandl.ApiConfig.api_key = "-rMNkitvuz-63uXCJpK9"

# In[]

# NASDAQOMX/OMXI8
# Start: 2014-07-01
# 8 actively traded shares on the Nasdaq OMX Iceland exchange

# NASDAQOMX/OMXIMCGI
# Start: 2008
# 11 companies
# Mid Cap companies listed on NASDAQ OMX Iceland Stock Exchange. 
# Share market value 150 million euro - 1 billion euro

# NASDAQOMX/OMXISCGI
# Start: 2008
# 5 companies
# Small Cap companies listed on NASDAQ OMX Iceland Stock Exchange. 
# Share market value <150 million euro

# NASDAQOMX/OMXIPI
# Start: 2008
# 17 companies
# All the shares listed on OMX Nordic Exchange Iceland. 
# Share market value <150 million euro

# NASDAQOMX/OMXIPI
# Start: 2008
# 269 companies ??
# all the listed companies traded on NASDAQ OMX First North. 


# In[]
omx ='NASDAQOMX/OMXI8' 
data = quandl.get(omx, start_date="2017-01-01", end_date="2017-01-10")
print(data)

