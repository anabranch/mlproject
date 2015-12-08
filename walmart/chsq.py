# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import pickle

# In[2]:

with open("data/train.pkl", 'rb') as f:
    train = pickle.load(f)

# In[3]:

trip_nums, X_dicts = zip(*train)
trip_nums = pd.Series(trip_nums)

# In[4]:

df = pd.read_csv("data/train.csv")

# In[5]:

trip_types = df[['TripType', 'VisitNumber']].groupby('VisitNumber').agg("mean")
len(trip_nums), len(trip_types)

# In[6]:

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# In[7]:

dv = DictVectorizer()
X = dv.fit_transform(list(X_dicts))
X.shape

# In[ ]:

support = SelectKBest(chi2, k=10000).fit(X, trip_nums.values)

# In[14]:

dv.restrict(support.get_support())

# In[15]:

X = transform(list(X_dicts))
X.shape

# In[ ]:
