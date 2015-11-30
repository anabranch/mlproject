import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer


def wrapStep(KH, step):
    name = step[0]
    start = NGMetricCheckPoint(KH, "validation", "start", name)
    end = NGMetricCheckPoint(KH, "validation", "end", name)
    return (name + "_start", start), step, (name + "_end", end)


class DataFrameToArray(TransformerMixin):
    def __init__(self):
        "Transform our DataFrame into an array"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.values


class NGMetricCheckPoint(TransformerMixin):
    def __init__(self, kagglehelper, validation_or_test, start_or_end,
                 metric_name="",
                 value="",
                 notes=""):
        "Records a start stop metric"
        self.kh = kagglehelper
        self.vot = validation_or_test
        self.soe = start_or_end
        self.m = metric_name
        self.v = value
        self.n = notes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.kh.record_metric(self.vot, self.soe, "in pipeline", self.m,
                              self.v, self.n)
        return X

    def get_params(self, deep):
        return dict(kagglehelper=self.kh,
                    validation_or_test=self.vot,
                    start_or_end=self.soe,
                    metric_name=self.m,
                    value=self.v,
                    notes=self.n)


class NGAddReturns(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df['Returns'] = df.ScanCount.map(lambda x: abs(x) if x < 0 else 0)
        df.ScanCount = df.ScanCount.map(lambda x: 0 if x < 0 else x)
        return df


class NGNAImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of strings are imputed with the most frequent value (mode)
        in column.
        Columns of other types are imputed with median of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else
                               X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


class GDummyAndKeepTransform(TransformerMixin):
    def __init__(self, cols_to_dummy, cols_to_keep, transformation_funcs):
        "Transform our DataFrame into an array"
        self.group_by_col = "VisitNumber"
        self.dummy_cols = cols_to_dummy
        self.keep_cols = cols_to_keep
        self.funcs = transformation_funcs

    def fit(self, X, y=None):
        self.vectorizer = DictVectorizer(sparse=False)
        self.vectorizer.fit(X[self.dummy_cols].T.to_dict().values())
        return self

    def transform(self, X, y=None):
        dummied = self.vectorizer \
                        .transform(X[self.dummy_cols].T.to_dict().values())

        nX = X[[self.group_by_col] + self.keep_cols]
        # print(nX)
        # print(pd.concat([nX, pd.DataFrame(dummied)], axis=1))
        return pd.concat([nX, pd.DataFrame(dummied)], axis=1) \
                 .groupby(self.group_by_col).agg(self.funcs)


class GMultiplierTransform(TransformerMixin):
    def __init__(self, one_hot_cols, mul_col):
        self.group_by_col = "VisitNumber"
        self.one_hot_cols = one_hot_cols
        self.mul_col = mul_col

    def fit(self, X, y=None):
        self.vectorizer = DictVectorizer(sparse=False)
        self.vectorizer.fit(X[self.one_hot_cols].T.to_dict().values())
        return self  # need to understand what's going on here

    def transform(self, X, y=None):
        if self.one_hot_cols:
            cols_vect = self.vectorizer.transform(
                X[self.one_hot_cols].T.to_dict().values())
            if self.mul_col:  # what is going on here?
                # why is this none always in your code?
                cols_vect = X[[self.mul_col]].as_matrix() * cols_vect
        return pd.concat([X[self.group_by_col], pd.DataFrame(cols_vect)], axis=1) \
                 .groupby(self.group_by_col).agg(np.sum)
