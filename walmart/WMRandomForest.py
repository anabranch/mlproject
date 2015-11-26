# Random Forest on DepartmentDescription Approach to Walmart Kaggle Competition

import pandas as pd
import numpy as np

from WalMartTransformer import WalmartImputer
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

#trans.fit(train_data, group_by_col = 'VisitNumber', one_hot_cols = ['DepartmentDescription'], mul_col = 'ScanCount', fillna = True)
train_matrix = trans.fit_transform(train_data, group_by_col = 'VisitNumber', one_hot_cols = ['DepartmentDescription'], mul_col = 'ScanCount', fillna = True)
test_matrix = test.transform(test_data)

############################Fit and Predict with Random Forest#################################

XTrain, XVal, yTrain, yVal = train_test_split(train_matrix, train_label, test_size=0.5, random_state=200)

search_grid = {
    "n_estimators": [100, 200, 300],
    "max_features": ["auto"],
    "min_samples_split": [10, 20, 5, 2],
    "min_samples_leaf": [1],
    "max_depth": [15, 30, 45, 60]
}

start = datetime.datetime.now()
clf = RandomForestClassifier()

model = GridSearchCV(clf, search_grid, n_jobs=3)
model.fit(XTrain, yTrain.ravel())
score = model.score(XVal, yVal.ravel())
print(score)

print("Total Time:", (datetime.datetime.now() - start).total_seconds())

#clf = DecisionTreeClassifier()
#clf.fit(train_matrix, train_label.ravel())
#pred_labels = clf.predict(test_matrix)

############################Output the prediction result#######################################

#predictions = pd.Series(pred_labels)
#output = pd.get_dummies(predictions)
#missing_categories = set(pd.DataFrame(train_label)[0]).difference(set(output.columns))
#for missing in missing_categories:
#    output[missing] = 0
#output = output[sorted(output.columns)]
#output.columns = ["TripType_%i" % x for x in output.columns]
#output = pd.concat([pd.Series(pd.DataFrame(test_before_gb).groupby(0).agg(np.sum).index.rename("VisitNumber")), output], axis=1).astype(int)
#output.to_csv("WMP.csv", index = False)


