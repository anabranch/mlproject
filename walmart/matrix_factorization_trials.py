import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition

import feature_transformers as ft
from kaggle_helper import KaggleHelper


def load_xy():
    raw = pd.read_csv("data/train.csv")
    y = raw[['TripType', 'VisitNumber']].groupby('VisitNumber').mean()
    X = raw.drop('TripType', axis=1)
    return X, y


def iterate_decomps():
    decompositions = [decomposition.PCA(), decomposition.NMF()]
    estimators = []
    for dc in decompositions:
        est = run_decomposition_pipeline(dc)
        estimators.append(est)


def run_decomposition_pipeline(decomp):
    kh = KaggleHelper("matrix_factorization.db")
    X, y = load_xy()

    ####### MUTABLE VARIABLES
    dummy_cols = ['Weekday', 'DepartmentDescription']
    keep_cols = ['VisitNumber', 'ScanCount', 'Returns']
    funcs = [np.sum, np.count_nonzero]
    logistic = LogisticRegression()
    decomp = decomp
    dfta = ft.DataFrameToArray()
    add_returns = ft.NGAddReturns()
    gdd = ft.GDummyAndKeepTransform(dummy_cols, keep_cols, funcs)

    kh.start_pipeline()
    transform_steps = list(
        ft.wrapStep(kh,
                    ("add_returns", add_returns)) + ft.wrapStep(kh,
                                                                ("gdd", gdd)))
    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)

    ####### END TRANSFORMATIONS
    X = transform_pipe.transform(X)

    ###### LATE STAGE VARIABLES
    feat_length = X.shape[1]
    n_components = np.linspace(10, feat_length, 5).astype(int)
    Cs = np.linspace(0.5, 10, 5)
    cv_grid = {'clf__C': Cs, 'decomp__n_components': n_components}
    num_folds = 4

    ####### START PREDICTIONS
    pred_steps = [('decomp', decomp), ('clf', logistic)]
    pred_pipe = Pipeline(steps=pred_steps)
    estimator = GridSearchCV(pred_pipe, cv_grid, cv=num_folds)

    kh.record_metric("validation", "start", estimator, "training", "", "")
    estimator.fit(X, y['TripType'].values)
    kh.record_metric("validation", "end", estimator, "training", "", "")

    pipeline_text = "Transformation Pipeline: " + str(transform_pipe)
    kh.record_metric("validation", "end", estimator, "best_params",
                     str(estimator.best_params_), pipeline_text)
    kh.record_metric("validation", "end", estimator, "best_estimator",
                     str(estimator.best_estimator_), pipeline_text)
    kh.record_metric("validation", "end", estimator, "best_score",
                     str(estimator.best_score_), pipeline_text)

    kh.end_pipeline()
    return estimator


if __name__ == '__main__':
    iterate_decomps()
