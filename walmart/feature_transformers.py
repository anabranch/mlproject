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
        print("X SHAPE:", str(X.shape))
        self.kh.record_metric(self.vot, self.soe, "in pipeline", "X Shape",
                              str(X.shape), "")
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
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            if col == "DepartmentDescription":
                X[col] = X[col].fillna("NAD")
            elif col == "FinelineNumber":
                X[col] = X[col].fillna("NAF")
            elif col == "Upc":
                X[col] = X[col].fillna("NAU")
            elif col == "Weekday":
                X[col] = X[col].fillna("NAW")
            elif col == "ScanCount":
                X[col] = X[col].fillna(0)
        return X


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
        return pd.concat([nX, pd.DataFrame(dummied)], axis=1) \
                 .groupby(self.group_by_col).agg(self.funcs)


class GMultiplierTransform(TransformerMixin):
    def __init__(self, one_hot_cols):
        self.group_by_col = "VisitNumber"
        self.one_hot_cols = one_hot_cols

    def fit(self, X, y=None):
        self.vectorizer = DictVectorizer(sparse=False)
        self.vectorizer.fit(X[self.one_hot_cols].T.to_dict().values())
        return self

    def transform(self, X, y=None):
        if self.one_hot_cols:
            cols_vect = self.vectorizer.transform(
                X[self.one_hot_cols].T.to_dict().values())
        return pd.concat([X[self.group_by_col], pd.DataFrame(cols_vect)], axis=1) \
                 .groupby(self.group_by_col).agg(np.sum)


class GDummyAndMultiplierTransform(TransformerMixin):
    def __init__(self, cols_to_dummy, mul_col):
        self.group_by_col = "VisitNumber"
        self.mul_col = mul_col
        self.dummy_cols = cols_to_dummy

    def fit(self, X, y=None):
        self.vectorizer = DictVectorizer(sparse=False)
        self.vectorizer.fit(X[self.dummy_cols].T.to_dict().values())
        return self

    def transform(self, X, y=None):
        cols_vect = self.vectorizer.transform(
            X[self.dummy_cols].T.to_dict().values())
        cols_vect = X[[self.mul_col]].as_matrix() * cols_vect
        return pd.concat([X[self.group_by_col], pd.DataFrame(cols_vect)], axis=1) \
                 .groupby(self.group_by_col).agg(np.sum)


class GDummyKeepAndMultiplierTransform(TransformerMixin):
    def __init__(self, cols_to_dummy, mul_col, keep_cols):
        self.group_by_col = "VisitNumber"
        self.mul_col = mul_col
        self.dummy_cols = cols_to_dummy
        self.keep_cols = keep_cols

    def fit(self, X, y=None):
        self.vectorizer = DictVectorizer(sparse=False)
        self.vectorizer.fit(X[self.dummy_cols].T.to_dict().values())
        print("1/2 way there!")
        self.keep_vectorizer = DictVectorizer(sparse=False)
        self.keep_vectorizer.fit(X[self.keep_cols].T.to_dict().values())
        return self

    def transform(self, X, y=None):
        print("transforming")
        cols_vect = self.vectorizer.transform(
            X[self.dummy_cols].T.to_dict().values())
        cols_vect = X[[self.mul_col]].as_matrix() * cols_vect

        print("completed dummy col vector")
        keep_vect = self.keep_vectorizer.transform(
            X[self.keep_cols].T.to_dict().values())
        print("completed keep col vector")

        X1 = pd.concat([X[self.group_by_col], pd.DataFrame(cols_vect)], axis=1) \
               .groupby(self.group_by_col).agg(np.sum)
        print("done grouping 1")

        X2 = pd.concat([X[self.group_by_col], pd.DataFrame(keep_vect)], axis=1) \
               .groupby(self.group_by_col).agg(np.mean)
        print("done grouping 2")
        print(X2.shape)
        print(X1.shape)

        return pd.concat([X1, X2], axis=1)
