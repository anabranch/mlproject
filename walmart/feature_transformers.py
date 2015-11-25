from sklearn.base import TransformerMixin


class DataFrameToArray(TransformerMixin):
    def __init__(self):
        "Transform our DataFrame into an array"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.values
