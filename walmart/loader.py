import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import feature_transformers as ft
from joblib import Memory
import pickle
import re

memory = Memory(cachedir='cached_funcs')


def positive_feature(key):
    if not key.endswith("D") \
       and not key.endswith("F") \
       and not key.startswith("U"):
        return True

    else:
        return False


def add_feature(sentence):
    feats = {}
    for k, v in sentence:
        if k.endswith("U"):
            if len(k) > 12:
                feats['U_lens_gt_16'] = 1
            else:
                feats['U_lens_lt_16'] = 1
    return feats

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

def filter_1(X_dicts):
    new_X = []
    for row in list(X_dicts):
        add_values = add_feature(row.items())
        row = {**row, **add_values}
        bad_keys = list(filter(positive_feature, row.keys()))
        for key in bad_keys:
            del row[key]
        new_X.append(row)
    print("done with filter 1")
    return new_X

def filter_2(new_X):
    NX = []
    for row in new_X:
        sentence = []
        negative_words = []
        for word, count in row.items():
            w1 = word.replace(",", "").replace(" ", "_")
            negative_word = re.search(r"-(\d+)_(.*)", w1)
            if negative_word:
                cnt = negative_word.group(1)
                g = negative_word.group(2)
                if len(g) > 1:
                    for x in range(int(cnt)):
                        negative_words.append(g)
            w = [w1]
            sentence += w * count
        for w in negative_words:
            try:
                sentence.remove(w)
            except ValueError:
                pass  # sometimes they aren't correct

        NX.append(sentence)
    print("done with filter 2")
    return NX


def load_xy2():
    with open("data/train.pkl", 'rb') as f:
        train = pickle.load(f)
    with open("data/test.pkl", 'rb') as f:
        test = pickle.load(f)

    train_trip_nums, train_X_dicts = zip(*train)
    df = pd.read_csv("data/train.csv")
    y = df[['TripType', 'VisitNumber']].groupby('VisitNumber').agg("mean")
    test_trip_nums, test_X_dicts = zip(*test)
    print("all loaded in")

    X = filter_2(filter_1(train_X_dicts))
    print("now onto test")

    X_test = pd.read_csv("data/test.csv",
                         dtype={'FinelineNumber': str,
                                'Weekday': str})

    X_test['Count'] = 1
    output_index = pd.Series(X_test[['VisitNumber', 'Count']] \
                             .groupby('VisitNumber').mean().index.tolist())
    
    print("starting with x_test")
    X_test = filter_2(filter_1(test_X_dicts))
    print("done with X_test")
    return X, y, X_test, output_index



def autosplit(func):
    def splitter(*args, **kwargs):
        val = func(*args, **kwargs)
        X = val['X']
        y = val['y']
        assert X.shape[1] == val['X_test'].shape[1]
        assert y.shape == (len(y), )

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

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
def XY1():
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
                      list(ft.wrapStep(("add_returns", add_returns))) + \
                      list(ft.wrapStep(('grouper', gdd))) + \
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
def XY2():  # Andy's Version
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['Weekday', 'DepartmentDescription']
    dfta = ft.DataFrameToArray()

    grouper = ft.GMultiplierTransform(dummy_cols)

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(('grouper', grouper)))

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
def XY3():  # Andy's Version
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['DepartmentDescription']
    mul_col = 'ScanCount'
    dfta = ft.DataFrameToArray()

    grouper = ft.GDummyAndMultiplierTransform(dummy_cols, mul_col)

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(('grouper', grouper)))

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
def XY4():  # Andy's Version
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['DepartmentDescription']
    keep_cols = ['Weekday']
    mul_col = 'ScanCount'
    dfta = ft.DataFrameToArray()

    grouper = ft.GDummyKeepAndMultiplierTransform(dummy_cols, mul_col,
                                                  keep_cols)

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(('grouper', grouper)))

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
def XY5():
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['FinelineNumber', 'DepartmentDescription']
    keep_cols = ['Weekday']
    mul_col = 'ScanCount'
    dfta = ft.DataFrameToArray()

    grouper = ft.GDummyKeepAndMultiplierTransform(dummy_cols, mul_col,
                                                  keep_cols)

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(('grouper', grouper)))

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
def XY6():
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['FinelineNumber', 'DepartmentDescription']
    keep_cols = ['Weekday']
    mul_col = 'ScanCount'
    dfta = ft.DataFrameToArray()

    grouper = ft.GDummyKeepAndMultiplierTransform(dummy_cols, mul_col,
                                                  keep_cols)

    transform_steps = list(ft.wrapStep(('grouper', grouper)))

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
def XY7():
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['DepartmentDescription']
    keep_cols = ['Weekday']
    mul_col = 'ScanCount'
    dfta = ft.DataFrameToArray()

    grouper = ft.GDummyKeepAndMultiplierTransform(dummy_cols, mul_col,
                                                  keep_cols)

    transform_steps = [("imputer", ft.NGNAImputer())] + \
                      list(ft.wrapStep(('grouper', grouper)))

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


@memory.cache
@autosplit
def XY8():
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['DepartmentDescription']
    keep_cols = ['Weekday', 'Returns']
    mul_col = 'ScanCount'
    dfta = ft.DataFrameToArray()
    add_returns = ft.NGAddReturns()

    grouper = ft.GDummyKeepAndMultiplierTransform(dummy_cols, mul_col,
                                                  keep_cols)

    transform_steps = [("imputer", ft.NGNAImputer()),
                       ("add_returns", add_returns), ('grouper', grouper)]

    ### DON'T CHANGE AFTER
    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)

    return {
        "X": transform_pipe.fit_transform(X),
        "y": y,
        "X_test": transform_pipe.transform(X_test),
        "X_test_index": X_test_index
    }


@autosplit
def XY9():
    X, y, X_test, X_test_index = load_xy()

    #### DON'T CHANGE BEFORE
    dummy_cols = ['FinelineNumber']
    keep_cols = ['Weekday', 'Returns']
    mul_col = None
    dfta = ft.DataFrameToArray()
    add_returns = ft.NGAddReturns()

    print("starting grouping")
    grouper = ft.GDummyKeepAndMultiplierTransform(dummy_cols, mul_col,
                                                  keep_cols)
    print("done grouping")
    transform_steps = [("imputer", ft.NGNAImputer()),
                       ("add_returns", add_returns), ('grouper', grouper)]

    ### DON'T CHANGE AFTER
    transform_steps.append((("dfta", dfta)))
    transform_pipe = Pipeline(steps=transform_steps)
    print("done with pipeline, now calculating")
    return {
        "X": transform_pipe.fit_transform(X),
        "y": y,
        "X_test": transform_pipe.transform(X_test),
        "X_test_index": X_test_index
    }



@autosplit
def XY10():
    with open('data/sentence_data.pkl', 'rb') as f:
        X, y, X_test, output_index = pickle.load(f)
        # this is basically just load_xy2 but cached

    print("transforming")
    X = [' '.join(q) for q in X]
    X_test = [' '.join(q) for q in X_test]
    print("tfidf")
    t = TfidfVectorizer(use_idf=False)
    X = t.fit_transform(X)
    print("for test")
    X_test = t.transform(X_test)
    print("returning")
    return {
        "X":X,
        "y":y.values.flatten(),
        "X_test":X_test,
        "X_test_index":output_index
    }
