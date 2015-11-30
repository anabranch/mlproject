# Random Forest on DepartmentDescription Approach to Walmart Kaggle Competition

import pandas as pd
import numpy as np

from WalMartTransformer import GWalmartTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import datetime

####################################PreProcessing##############################################

# Read in the train and test data.
raw = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")

# Groupby TripType to find train labels
train_label = raw[['TripType', 'VisitNumber']]
train_label = train_label.groupby('VisitNumber').agg(np.mean).as_matrix()

# Fill the NAs for train and test data
train_data = raw[raw_test.columns]
test_data = raw_test

trans = GWalmartTransformer()

trans.fit(train_data, group_by_col = 'VisitNumber', one_hot_cols = ['DepartmentDescription'], mul_col = 'ScanCount', fillna = True)
train_matrix = trans.transform(train_data)
test_matrix = trans.transform(test_data)

print(train_matrix)
print(test_matrix)

print(train_matrix.shape)
print(test_matrix.shape)
############################Fit and Predict with Random Forest#################################
