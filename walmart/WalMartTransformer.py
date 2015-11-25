# This file contains transformers of walmart kaggle competition data.
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

class WalmartImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of strings are imputed with the most frequent value (mode)
        in column.
        Columns of other types are imputed with median of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

class WalmartOneHotEncoder(TransformerMixin):
    def __init__(self):
        """One hot encoder for one column."""

    def fit(self, X, y=None, column_name = 'DepartmentDescription'):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)