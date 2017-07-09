# -*- coding: utf-8 -*-
# In[]
from yahoo_finance import Share


# In[]
ossur = Share("OSSR.CO")
x = ossur.get_historical('2014-04-25', '2014-04-29')
print(x)

# In[]

