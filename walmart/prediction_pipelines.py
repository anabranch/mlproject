import argparse

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition

from kaggle_helper import KaggleHelper
import feature_transformers as ft
import loader
import utils

KH = KaggleHelper("matrix_factorization.db")


def iterate_decomps():
    decompositions = [decomposition.TruncatedSVD(), decomposition.NMF()]
    estimators = []
    for dc in decompositions:
        est = run_decomposition_pipeline(dc)
        estimators.append(est)


def run_decomposition_pipeline(decomp):
    ###### DATA LOADING
    xy = loader.XY5(KH)  # CAN CHANGE

    X = xy['X_train']
    y = xy['y_train']
    X_val = xy['X_val']
    y_val = xy['y_val']
    X_test = xy['X_test']
    output_index = xy['X_test_index']
    print("LOADED DATA")

    ###### PIPELINE/CV VARIABLES
    ###### DO NOT CHANGE BEFORE
    clf = LinearSVC()
    decomp = decomp
    fl = X.shape[1]  # use for n_components
    cv_grid = {
        'clf__C': np.linspace(0.1, 10, 8),
        'decomp__n_components': np.linspace(int(fl / 2), fl, 3).astype(int)
    }
    num_folds = 3

    ####### START PREDICTIONS
    print("TRAINING ESTIMATOR")
    pred_pipe = Pipeline(steps=[('decomp', decomp), ('clf', clf)])

    ###### DO NOT CHANGE AFTER
    estimator = GridSearchCV(pred_pipe, cv_grid, cv=num_folds)

    # DO NOT NEED TO CHANGE BEYOND THIS LINE
    KH.record_metric("validation", "start", estimator, "training", "", "")
    estimator.fit(X, y)
    KH.record_metric("validation", "end", estimator, "training", "", "")
    KH.record_metric("validation", "end", estimator, "best_params",
                     str(estimator.best_params_), "NA")
    KH.record_metric("validation", "end", estimator, "best_estimator",
                     str(estimator.best_estimator_), "NA")
    KH.record_metric("validation", "end", estimator, "best_score",
                     str(estimator.best_score_), "NA")
    validation_score = str(estimator.score(X_val, y_val))
    KH.record_metric("validation", "end", estimator, "validation score",
                     validation_score, "")

    preds = estimator.predict(X_test)
    predictions = pd.DataFrame(
        {"VisitNumber": output_index,
         "TripType": preds})
    KH.save_test_predictions(utils.convert_predictions(predictions), estimator,
                             "predictions")
    KH.end_pipeline()

    return estimator


def run_logistic_pipeline():
    ###### DATA LOADING
    xy = loader.XY2(KH)  # CAN CHANGE

    X = xy['X_train']
    y = xy['y_train']
    X_val = xy['X_val']
    y_val = xy['y_val']
    X_test = xy['X_test']
    output_index = xy['X_test_index']
    print("LOADED DATA")

    ###### PIPELINE/CV VARIABLES
    ###### DO NOT CHANGE BEFORE
    clf = LogisticRegression()
    fl = X.shape[1]  # use for n_components
    cv_grid = {'clf__C': np.linspace(0, 10, 8), }
    num_folds = 3

    ####### START PREDICTIONS
    print("TRAINING ESTIMATOR")
    pred_pipe = Pipeline(steps=[('clf', clf)])

    ###### DO NOT CHANGE AFTER
    estimator = GridSearchCV(pred_pipe, cv_grid, cv=num_folds)

    # DO NOT NEED TO CHANGE BEYOND THIS LINE
    KH.record_metric("validation", "start", estimator, "training", "", "")
    estimator.fit(X, y)
    KH.record_metric("validation", "end", estimator, "training", "", "")
    KH.record_metric("validation", "end", estimator, "best_params",
                     str(estimator.best_params_), "NA")
    KH.record_metric("validation", "end", estimator, "best_estimator",
                     str(estimator.best_estimator_), "NA")
    KH.record_metric("validation", "end", estimator, "best_score",
                     str(estimator.best_score_), "NA")
    validation_score = str(estimator.score(X_val, y_val))
    KH.record_metric("validation", "end", estimator, "validation score",
                     validation_score, "")

    preds = estimator.predict(X_test)
    predictions = pd.DataFrame(
        {"VisitNumber": output_index,
         "TripType": preds})
    KH.save_test_predictions(utils.convert_predictions(predictions), estimator,
                             "predictions")
    KH.end_pipeline()

    return estimator


def run_svc_pipeline():
    ###### DATA LOADING
    xy = loader.XY2(KH)  # CAN CHANGE

    X = xy['X_train']
    y = xy['y_train']
    X_val = xy['X_val']
    y_val = xy['y_val']
    X_test = xy['X_test']
    output_index = xy['X_test_index']
    print("LOADED DATA")

    ###### PIPELINE/CV VARIABLES
    ###### DO NOT CHANGE BEFORE
    clf = LinearSVC()
    fl = X.shape[1]  # use for n_components
    cv_grid = {'clf__C': np.linspace(0, 10, 8), }
    num_folds = 3

    ####### START PREDICTIONS
    print("TRAINING ESTIMATOR")
    pred_pipe = Pipeline(steps=[('clf', clf)])

    ###### DO NOT CHANGE AFTER
    estimator = GridSearchCV(pred_pipe, cv_grid, cv=num_folds)

    # DO NOT NEED TO CHANGE BEYOND THIS LINE
    KH.record_metric("validation", "start", estimator, "training", "", "")
    estimator.fit(X, y)
    KH.record_metric("validation", "end", estimator, "training", "", "")
    KH.record_metric("validation", "end", estimator, "best_params",
                     str(estimator.best_params_), "NA")
    KH.record_metric("validation", "end", estimator, "best_estimator",
                     str(estimator.best_estimator_), "NA")
    KH.record_metric("validation", "end", estimator, "best_score",
                     str(estimator.best_score_), "NA")
    validation_score = str(estimator.score(X_val, y_val))
    KH.record_metric("validation", "end", estimator, "validation score",
                     validation_score, "")

    preds = estimator.predict(X_test)
    predictions = pd.DataFrame(
        {"VisitNumber": output_index,
         "TripType": preds})
    KH.save_test_predictions(utils.convert_predictions(predictions), estimator,
                             "predictions")
    KH.end_pipeline()

    return estimator


def run_random_forest_pipeline():
    ###### DATA LOADING
    xy = loader.XY2(KH)  # CAN CHANGE

    X = xy['X_train']
    y = xy['y_train']
    X_val = xy['X_val']
    y_val = xy['y_val']
    X_test = xy['X_test']
    output_index = xy['X_test_index']
    print("LOADED DATA")

    ###### PIPELINE/CV VARIABLES
    ###### DO NOT CHANGE BEFORE
    clf = RandomForestClassifier()
    fl = X.shape[1]  # use for n_components
    cv_grid = {
        "clf__n_estimators": [100],
        "clf__max_features": ["auto"],
        "clf__min_samples_split": [10],
        "clf__min_samples_leaf": [1],
        "clf__max_depth": [None]
    }
    num_folds = 3

    ####### START PREDICTIONS
    print("TRAINING ESTIMATOR")
    pred_pipe = Pipeline(steps=[('clf', clf)])

    ###### DO NOT CHANGE AFTER
    estimator = GridSearchCV(pred_pipe, cv_grid, cv=num_folds)

    # DO NOT NEED TO CHANGE BEYOND THIS LINE
    KH.record_metric("validation", "start", estimator, "training", "", "")
    estimator.fit(X, y)
    KH.record_metric("validation", "end", estimator, "training", "", "")
    KH.record_metric("validation", "end", estimator, "best_params",
                     str(estimator.best_params_), "NA")
    KH.record_metric("validation", "end", estimator, "best_estimator",
                     str(estimator.best_estimator_), "NA")
    KH.record_metric("validation", "end", estimator, "best_score",
                     str(estimator.best_score_), "NA")
    validation_score = str(estimator.score(X_val, y_val))
    KH.record_metric("validation", "end", estimator, "validation score",
                     validation_score, "")

    preds = estimator.predict(X_test)
    predictions = pd.DataFrame(
        {"VisitNumber": output_index,
         "TripType": preds})
    KH.save_test_predictions(utils.convert_predictions(predictions), estimator,
                             "predictions")
    KH.end_pipeline()

    return estimator


def run_knn_pipeline():
    ###### DATA LOADING
    xy = loader.XY4(KH)  # CAN CHANGE

    X = xy['X_train']
    y = xy['y_train']
    X_val = xy['X_val']
    y_val = xy['y_val']
    X_test = xy['X_test']
    output_index = xy['X_test_index']
    print("LOADED DATA")

    ###### PIPELINE/CV VARIABLES
    ###### DO NOT CHANGE BEFORE
    clf = KNeighborsClassifier()
    fl = X.shape[1]  # use for n_components
    cv_grid = {"clf__metric": ['euclidean', 'manhattan']}
    num_folds = 3

    ####### START PREDICTIONS
    print("TRAINING ESTIMATOR")
    pred_pipe = Pipeline(steps=[('clf', clf)])

    ###### DO NOT CHANGE AFTER
    estimator = GridSearchCV(pred_pipe, cv_grid, cv=num_folds)

    # DO NOT NEED TO CHANGE BEYOND THIS LINE
    KH.record_metric("validation", "start", estimator, "training", "", "")
    estimator.fit(X, y)
    KH.record_metric("validation", "end", estimator, "training", "", "")
    KH.record_metric("validation", "end", estimator, "best_params",
                     str(estimator.best_params_), "NA")
    KH.record_metric("validation", "end", estimator, "best_estimator",
                     str(estimator.best_estimator_), "NA")
    KH.record_metric("validation", "end", estimator, "best_score",
                     str(estimator.best_score_), "NA")
    validation_score = str(estimator.score(X_val, y_val))
    KH.record_metric("validation", "end", estimator, "validation score",
                     validation_score, "")

    preds = estimator.predict(X_test)
    predictions = pd.DataFrame(
        {"VisitNumber": output_index,
         "TripType": preds})
    KH.save_test_predictions(utils.convert_predictions(predictions), estimator,
                             "predictions")
    KH.end_pipeline()

    return estimator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pipeline')
    args = parser.parse_args()
    if args.pipeline == 'knn':
        run_knn_pipeline()
    elif args.pipeline == 'rf':
        run_random_forest_pipeline()
    elif args.pipeline == 'logistic':
        run_logistic_pipeline()
    elif args.pipeline == 'svc':
        run_svc_pipeline()
    elif args.pipeline == 'decomps':
        iterate_decomps()
