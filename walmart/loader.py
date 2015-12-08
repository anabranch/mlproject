import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
import feature_transformers as ft


def load_xy():
    raw = pd.read_csv("data/train.csv",
                      dtype={'FinelineNumber': str,
                             'Weekday': str})

    y = raw[['TripType', 'VisitNumber']] \
        .groupby('VisitNumber').mean()['TripType'].values
    X = raw.drop('TripType', axis=1)

    X_test = pd.read_csv("data/test.csv",
                         dtype={'FinelineNumber': str,
                                'Weekday': str})

    X_test['Count'] = 1
    output_index = pd.Series(X_test[['VisitNumber', 'Count']] \
                             .groupby('VisitNumber').mean().index.tolist())

    return X, y, X_test.drop("Count", axis=1), output_index


def autosplit(func):
    def splitter(*args, **kwargs):
        val = func(*args, **kwargs)
        X = val['X']
        y = val['y']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01)

        assert X.shape[1] == val['X_test'].shape[1]
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": val['X_test'],
            "X_test_index": val['X_test_index']
        }

    return splitter


@autosplit
def XY1(kh):
    X, y, X_test, X_test_index = load_xy()

    ####### VARIABLES
    dummy_cols = ['Weekday', 'DepartmentDescription']
    keep_cols = ['ScanCount', 'Returns']
    funcs = [np.sum, np.count_nonzero]

    dfta = ft.DataFrameToArray()
    add_returns = ft.NGAddReturns()
    gdd = ft.GDummyAndKeepTransform(dummy_cols, keep_cols,
                                    funcs)  # Doesn't work!

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(kh, ("add_returns", add_returns))) + \
                      list(ft.wrapStep(kh, ('grouper', gdd))) + \
                      [("dfta", dfta)]
    transform_pipe = Pipeline(steps=transform_steps)

    kh.start_pipeline()
    kh.record_metric("validation", "start", "NA", "transform_pipeline",
                     str(transform_pipe), "NA")

    return {
        "X": transform_pipe.fit_transform(X),
        "y": y,
        "X_test": transform_pipe.transform(X_test),
        "X_test_index": X_test_index
    }


@autosplit
def XY2(kh):  # Andy's Version
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['Weekday', 'DepartmentDescription']
    dfta = ft.DataFrameToArray()

    grouper = ft.GMultiplierTransform(dummy_cols)

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(kh, ('grouper', grouper)))

    ### DON'T CHANGE AFTER
    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)

    kh.start_pipeline()
    kh.record_metric("validation", "start", "NA", "transform_pipeline",
                     str(transform_pipe), "NA")

    return {
        "X": transform_pipe.fit_transform(X),
        "y": y,
        "X_test": transform_pipe.transform(X_test),
        "X_test_index": X_test_index
    }


@autosplit
def XY3(kh):  # Andy's Version
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['DepartmentDescription']
    mul_col = 'ScanCount'
    dfta = ft.DataFrameToArray()

    grouper = ft.GDummyAndMultiplierTransform(dummy_cols, mul_col)

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(kh, ('grouper', grouper)))

    ### DON'T CHANGE AFTER
    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)

    kh.start_pipeline()
    kh.record_metric("validation", "start", "NA", "transform_pipeline",
                     str(transform_pipe), "NA")

    return {
        "X": transform_pipe.fit_transform(X),
        "y": y,
        "X_test": transform_pipe.transform(X_test),
        "X_test_index": X_test_index
    }


@autosplit
def XY4(kh):  # Andy's Version
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['DepartmentDescription']
    keep_cols = ['Weekday']
    mul_col = 'ScanCount'
    dfta = ft.DataFrameToArray()

    grouper = ft.GDummyKeepAndMultiplierTransform(dummy_cols, mul_col,
                                                  keep_cols)

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(kh, ('grouper', grouper)))

    ### DON'T CHANGE AFTER
    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)

    kh.start_pipeline()
    kh.record_metric("validation", "start", "NA", "transform_pipeline",
                     str(transform_pipe), "NA")

    return {
        "X": transform_pipe.fit_transform(X),
        "y": y,
        "X_test": transform_pipe.transform(X_test),
        "X_test_index": X_test_index
    }


@autosplit
def XY5(kh):
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['FinelineNumber', 'DepartmentDescription']
    keep_cols = ['Weekday']
    mul_col = 'ScanCount'
    dfta = ft.DataFrameToArray()

    grouper = ft.GDummyKeepAndMultiplierTransform(dummy_cols, mul_col,
                                                  keep_cols)

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(kh, ('grouper', grouper)))

    ### DON'T CHANGE AFTER
    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)

    kh.start_pipeline()
    kh.record_metric("validation", "start", "NA", "transform_pipeline",
                     str(transform_pipe), "NA")

    return {
        "X": transform_pipe.fit_transform(X),
        "y": y,
        "X_test": transform_pipe.transform(X_test),
        "X_test_index": X_test_index
    }
