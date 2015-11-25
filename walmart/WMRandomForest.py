# Random Forest on DepartmentDescription Approach to Walmart Kaggle Competition

import pandas as pd
import numpy as np

from WalMartTransformer import WalmartImputer
from sklearn.feature_extraction import DictVectorizer

####################################PreProcessing##############################################

# Read in the train and test data.
raw = pd.read_csv("data/train.csv")
raw_test = pd.read_csv("data/test.csv")

# Groupby TripType to find train labels
train_label = raw[['TripType', 'VisitNumber']]
train_label = train_label.groupby('VisitNumber').agg(np.mean).as_matrix()

# Fill the NAs for train and test data
imp = WalmartImputer()
train_data = raw[['VisitNumber', 'ScanCount', 'DepartmentDescription']]
test_data = raw_test[['VisitNumber', 'ScanCount', 'DepartmentDescription']]
train_data = imp.fit_transform(train_data)
test_data = imp.transform(test_data)

# One hot transformation for train and test DepartmentDescription
train_data_dept = train_data[['DepartmentDescription']]
test_data_dept = test_data[['DepartmentDescription']]
vectorizer = DictVectorizer(sparse = False)
train_data_dept_vect = vectorizer.fit_transform(train_data_dept.T.to_dict().values())
test_data_dept_vect = vectorizer.transform(test_data_dept.T.to_dict().values())

print(train_data_dept_vect)
print(test_data_dept_vect)



