import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
import feature_transformers as ft


def load_xy():
    raw = pd.read_csv("data/train.csv")
    y = raw[['TripType', 'VisitNumber']] \
        .groupby('VisitNumber').mean()['TripType'].values
    X = raw.drop('TripType', axis=1)
    X_test = pd.read_csv("data/test.csv")
    return X, y, X_test


def autosplit(func):
    def splitter(*args, **kwargs):
        val = func(*args, **kwargs)
        X = val['X']
        y = val['y']
        X_train, X_val, y_train, y_val = train_test_split(X, y)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": val['X_test'],
            "trip_types": pd.Series(y).unique(),
            "X_test_index": val['X_test_index']
        }

    return splitter


@autosplit
def XY1(kh):  # Dumb Version
    X, y, X_test = load_xy()

    ####### VARIABLES
    dummy_cols = ['Weekday', 'DepartmentDescription']
    keep_cols = ['VisitNumber', 'ScanCount', 'Returns']
    funcs = [np.sum, np.count_nonzero]

    dfta = ft.DataFrameToArray()
    add_returns = ft.NGAddReturns()
    gdd = ft.GDummyAndKeepTransform(dummy_cols, keep_cols, funcs)
    transform_steps = list(ft.wrapStep(kh, ("add_returns", add_returns)) \
                           + ft.wrapStep(kh, ("gdd", gdd)))

    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)

    kh.start_pipeline()
    kh.record_metric("validation", "start", "NA", "transform_pipeline",
                     str(transform_pipe), "NA")

    return {
        "X": transform_pipe.transform(X),
        "y": y,
        "X_test": transform_pipe.transform(X_test),
        "X_test_index": pd.Series(X_test.index)
    }


@autosplit
def XY2(kh):  # Andy's Version
    X, y, X_test = load_xy()

    dummy_cols = ['Weekday', 'DepartmentDescription']
    dfta = ft.DataFrameToArray()

    grouper = ft.GWalmartTransformer(dummy_cols, None)

    transform_steps = list(ft.wrapStep(kh, ('grouper', grouper)))
    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)

    kh.start_pipeline()
    kh.record_metric("validation", "start", "NA", "transform_pipeline",
                     str(transform_pipe), "NA")

    return {
        "X": transform_pipe.transform(X),
        "y": y,
        "X_test": transform_pipe.transform(X_test),
        "X_test_index": pd.Series(X_test.index)
    }
