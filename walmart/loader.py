import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

import feature_transformers as ft


def load_xy():
    raw = pd.read_csv("data/train.csv")
    y = raw[['TripType', 'VisitNumber']].groupby('VisitNumber').mean()
    X = raw.drop('TripType', axis=1)
    XTest = pd.read_csv("data/test.csv")
    return X, y, XTest


def XY1(kh):  # Dumb Version
    X, y, XTest = load_xy()

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
    X = transform_pipe.transform(X)
    XTest = transform_pipe.transform(XTest)

    return {"X": X, "y": y, "XTest": XTest}


def XY2(kh):  # Andy's Version
    X, y, XTest = load_xy()

    dummy_cols = ['Weekday', 'DepartmentDescription']
    dfta = ft.DataFrameToArray()

    grouper = ft.GWalmartTransformer(dummy_cols, None)

    transform_steps = list(ft.wrapStep(kh, ('grouper', grouper)))
    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)

    kh.start_pipeline()
    kh.record_metric("validation", "start", "NA", "transform_pipeline",
                     str(transform_pipe), "NA")

    X = transform_pipe.fit_transform(X)
    XTest = transform_pipe.transform(XTest)

    return {"X": X, "y": y, "XTest": XTest}
