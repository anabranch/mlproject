from sklearn.base import TransformerMixin


class DataFrameToArray(TransformerMixin):
    def __init__(self):
        "Transform our DataFrame into an array"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.values


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


class GDeptDescriptionWithTransform(TransformerMixin):
    def __init__(self, cols_to_dummy, cols_to_keep, transformation_funcs):
        "Transform our DataFrame into an array"
        self.dummy_cols = cols_to_dummy
        self.keep_cols = cols_to_keep
        self.funcs = transformation_funcs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        col_list = df.columns.tolist()
        convert_column_functions = {x: self.funcs for x in col_list}

        dummied = pd.get_dummies(df[self.dummy_cols])

        return pd.concat([df[self.keep_cols], dummied], axis=1) \
                 .groupby('VisitNumber') \
                 .agg(convert_column_functions)
