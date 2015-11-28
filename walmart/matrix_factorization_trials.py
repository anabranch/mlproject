import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition

import feature_transformers as ft
import loader
from kaggle_helper import KaggleHelper


def iterate_decomps():
    decompositions = [decomposition.PCA(), decomposition.NMF()]
    estimators = []
    for dc in decompositions:
        est = run_decomposition_pipeline(dc)
        estimators.append(est)


def run_decomposition_pipeline(decomp):
    kh = KaggleHelper("matrix_factorization.db")
    clf = LogisticRegression()
    decomp = decomp

    xy = loader.XY1(kh)
    X = xy['X']
    y = xy['y']

    ###### PIPELINE/CV VARIABLES
    feat_length = X.shape[1]  # use for n_components
    cv_grid = {
        'clf__C': np.linspace(0.5, 10, 3),
        'decomp__n_components': np.linspace(10, feat_length, 3).astype(int)
    }
    num_folds = 4

    ####### START PREDICTIONS
    pred_pipe = Pipeline(steps=[('decomp', decomp), ('clf', clf)])
    estimator = GridSearchCV(pred_pipe, cv_grid, cv=num_folds)

    kh.record_metric("validation", "start", estimator, "training", "", "")
    estimator.fit(X, y['TripType'].values)

    kh.record_metric("validation", "end", estimator, "training", "", "")
    kh.record_metric("validation", "end", estimator, "best_params",
                     str(estimator.best_params_), "NA")
    kh.record_metric("validation", "end", estimator, "best_estimator",
                     str(estimator.best_estimator_), "NA")
    kh.record_metric("validation", "end", estimator, "best_score",
                     str(estimator.best_score_), "NA")
    kh.end_pipeline()
    return estimator


if __name__ == '__main__':
    iterate_decomps()
