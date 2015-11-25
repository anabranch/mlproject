# Random Forest on DepartmentDescription Approach to Walmart Kaggle Competition

import pandas as pd
import numpy as np

from WalMartTransformer import WalmartImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

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
train_dept_vect = vectorizer.fit_transform(train_data_dept.T.to_dict().values())
test_dept_vect = vectorizer.transform(test_data_dept.T.to_dict().values())

# Fill scan count into the one hot representation 
train_dept_scanCount = train_data[['ScanCount']].as_matrix() * train_dept_vect
test_dept_scanCount = test_data[['ScanCount']].as_matrix() * test_dept_vect

train_before_gb = np.hstack((train_data[['VisitNumber']].as_matrix(), train_dept_scanCount))
test_before_gb = np.hstack((test_data[['VisitNumber']].as_matrix(), test_dept_scanCount))

train_matrix = pd.DataFrame(train_before_gb).groupby(0).agg(np.sum).as_matrix()
test_matrix = pd.DataFrame(test_before_gb).groupby(0).agg(np.sum).as_matrix()

############################Fit and Predict with Random Forest#################################

clf = RandomForestClassifier(n_estimators = 250, n_jobs = 3, random_state = 42, criterion='entropy', min_samples_split = 4, max_features = None)
clf.fit(train_matrix, train_label.ravel())
pred_labels = clf.predict(test_matrix)

############################Output the prediction result#######################################

predictions = pd.Series(pred_labels)
output = pd.get_dummies(predictions)
missing_categories = set(pd.DataFrame(train_label)[0]).difference(set(output.columns))
for missing in missing_categories:
    output[missing] = 0
output = output[sorted(output.columns)]
output.columns = ["TripType_%i" % x for x in output.columns]
output = pd.concat([pd.Series(pd.DataFrame(test_before_gb).groupby(0).agg(np.sum).index.rename("VisitNumber")), output], axis=1).astype(int)
output.to_csv("WMP.csv", index = False)


