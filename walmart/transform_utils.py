import pandas as pd
import numpy as np


def preprocess_walmart(df, cols_to_transform, keep_cols, train=False):
    df['Returns'] = df.ScanCount.map(lambda x: abs(x) if x < 0 else 0)
    df.ScanCount = df.ScanCount.map(lambda x: 0 if x < 0 else x)
    if train:
        keep_cols.append("TripType")
    df = transform_column(df, cols_to_transform, keep_cols)
    return transform_group(df, 'VisitNumber', train)


def transform_column(df, colname, keep_columns):
    dummied = pd.get_dummies(df[colname])
    return pd.concat([df[keep_columns], dummied], axis=1)


def generate_convert_dict(col_list, func, train=False):
    convert_dict = {x: func for x in col_list}
    return convert_dict


def transform_group(df, groupby_col, train=False):
    col_list = df.columns.tolist()
    f1 = generate_convert_dict(col_list, np.sum, train)
    f2 = generate_convert_dict(col_list, np.count_nonzero, train)
    if train: f1['TripType'] = np.mean
    first = df.groupby(groupby_col).agg(f1)
    second = df.groupby(groupby_col).agg(f2)
    return pd.concat([first, second], axis=1)


def convert_predictions(predictions, **kwargs):
    predictions = pd.Series(predictions)
    output_index = kwargs['output_index']
    actual_trip_types = kwargs['actual_trip_types']
    output = pd.get_dummies(predictions)
    missing_categories = set(actual_trip_types).difference(set(output.columns))
    for missing in missing_categories:
        output[missing] = 0
    print("Missing Categories", missing_categories)
    output = output[sorted(output.columns)]
    output.columns = ["TripType_%i" % x for x in output.columns]
    return pd.concat([pd.Series(output_index), output], axis=1).astype(int)
