# This file contains transformers of walmart kaggle competition data.
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer

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

class GWalmartTransformer(TransformerMixin):
    def __init__(self, group_by_col, one_hot_cols, mul_col, fillna):
        self.fillna = fillna
        self.group_by_col = group_by_col
        self.one_hot_cols = one_hot_cols
        self.mul_col = mul_col

    def fit(self, X, y = None):
        if self.fillna:
            self.imp = WalmartImputer()
            self.imp.fit(X)
        if self.one_hot_cols:
            self.vectorizer = DictVectorizer(sparse = False)
            self.vectorizer.fit(X[self.one_hot_cols].T.to_dict().values())
        return self

    def transform(self, X, y = None):
        if self.fillna:
            self.imp.transform(X)
        if self.one_hot_cols:
            cols_vect = self.vectorizer.transform(X[self.one_hot_cols].T.to_dict().values())
            if self.mul_col:
                cols_vect = X[[self.mul_col]].as_matrix() * cols_vect
        return pd.concat([X[self.group_by_col], pd.DataFrame(cols_vect)], axis = 1).groupby(self.group_by_col).agg(np.sum).as_matrix()
