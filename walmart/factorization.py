import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition

from kaggle_helper import KaggleHelper
import feature_transformers as ft
import loader
import utils


def iterate_decomps():
    decompositions = [decomposition.PCA()]
    estimators = []
    for dc in decompositions:
        est = run_decomposition_pipeline(dc)
        estimators.append(est)


def run_decomposition_pipeline(decomp):
    kh = KaggleHelper("matrix_factorization.db")
    ###### DATA LOADING
    xy = loader.XY2(kh)
    X = xy['X_train']
    y = xy['y_train']
    X_val = xy['X_val']
    y_val = xy['y_val']
    X_test = xy['X_test']
    output_index = xy['X_test_index']

    print("LOADED DATA")

    ###### PIPELINE/CV VARIABLES
    #    clf = LinearSVC()
    clf = LogisticRegression()
    decomp = decomp
    fl = X.shape[1]  # use for n_components
    num_d = 1
    cv_grid = {
        #        'clf__C': np.linspace(0.5, 10, num_d),
        'decomp__n_components': np.linspace(10, fl, num_d).astype(int)
    }
    num_folds = 4

    ####### START PREDICTIONS
    print("TRAINING ESTIMATOR")
    pred_pipe = Pipeline(steps=[('decomp', decomp), ('clf', clf)])
    estimator = GridSearchCV(pred_pipe, cv_grid, cv=num_folds)

    # DO NOT NEED TO CHANGE BEYOND THIS LINE
    kh.record_metric("validation", "start", estimator, "training", "", "")
    estimator.fit(X, y)
    kh.record_metric("validation", "end", estimator, "training", "", "")
    kh.record_metric("validation", "end", estimator, "best_params",
                     str(estimator.best_params_), "NA")
    kh.record_metric("validation", "end", estimator, "best_estimator",
                     str(estimator.best_estimator_), "NA")
    kh.record_metric("validation", "end", estimator, "best_score",
                     str(estimator.best_score_), "NA")
    validation_score = str(estimator.score(X_val, y_val))
    kh.record_metric("validation", "end", estimator, "validation score",
                     validation_score, "")

    preds = estimator.predict(X_test)
    predictions = pd.DataFrame(
        {"VisitNumber": output_index,
         "TripType": preds})
    kh.save_test_predictions(utils.convert_predictions(predictions), estimator,
                             "predictions")
    kh.end_pipeline()

    return estimator


if __name__ == '__main__':
    iterate_decomps()
