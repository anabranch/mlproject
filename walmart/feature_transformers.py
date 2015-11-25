import pandas as pd
from sklearn.base import TransformerMixin


def generateMetricTransformPair(KH, name):
    start = NGMetricCheckPoint(KH, "validation", "start", name, "", "")
    end = NGMetricCheckPoint(KH, "validation", "stop", name, "", "")
    return start, end


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


class GDummyAndKeepTransform(TransformerMixin):
    def __init__(self, cols_to_dummy, cols_to_keep, transformation_funcs):
        "Transform our DataFrame into an array"
        self.dummy_cols = cols_to_dummy
        self.keep_cols = cols_to_keep
        self.funcs = transformation_funcs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        dummied = pd.get_dummies(df[self.dummy_cols])

        col_list = dummied.columns.tolist() + self.keep_cols
        convert_column_functions = {x: self.funcs for x in col_list}

        output = pd.concat([df[self.keep_cols], dummied], axis=1) \
                 .groupby('VisitNumber') \
                 .agg(convert_column_functions)

        return output
