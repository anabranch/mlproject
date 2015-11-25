import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model, decomposition

import feature_transformers as ft
from kaggle_helper import KaggleHelper


def load_xy():
    raw = pd.read_csv("data/train.csv")
    print("Unique UPC:", len(raw.Upc.unique()))
    print("Unique DepartmentDescription:",
          len(raw.DepartmentDescription.unique()))
    print("Unique FinelineNumber:", len(raw.FinelineNumber.unique()))
    y = raw[['TripType', 'VisitNumber']].groupby('VisitNumber').mean()
    X = raw.drop('TripType', axis=1)
    return X, y


def run_pca_pipeline():
    kh = KaggleHelper("matrix_factorization.db")
    X, y = load_xy()

    # Mutable vals
    dummy_cols = ['Weekday', 'DepartmentDescription']
    keep_cols = ['VisitNumber', 'ScanCount', 'Returns']
    funcs = [np.sum, np.count_nonzero]
    n_components = [10, 20, 40, 80, 160]
    Cs = np.logspace(-4, 4, 3)
    cv_grid = {'logistic__C': Cs, 'pca__n_components': n_components}
    num_folds = 6

    logistic = linear_model.LogisticRegression()
    pca = decomposition.PCA()
    dfta = ft.DataFrameToArray()
    add_returns = ft.NGAddReturns()
    gdd = ft.GDummyAndKeepTransform(dummy_cols, keep_cols, funcs)
    ar1, ar2 = ft.generateMetricTransformPair(kh, "add_returns")
    gdd1, gdd2 = ft.generateMetricTransformPair(kh, "grouping + dummy")

    kh.start_pipeline()
    steps = [('add_returns_start', ar1), ('add_returns', add_returns),
             ('add_returns_end', ar2), ('gdd_returns_start', gdd1),
             ('gdd', gdd), ('gdd_returns_start', gdd2), ('dftoarr', dfta),
             ('pca', pca), ('logistic', logistic)]
    pipe = Pipeline(steps=steps)

    estimator = GridSearchCV(pipe, cv_grid, cv=num_folds)
    estimator.fit(X, y)
    kh.end_pipeline()
    return estimator
