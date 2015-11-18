import pandas as pd
import numpy as np


def preprocess_walmart(df):
    return df
    # df['Returns'] = df.ScanCount.map(lambda x: abs(x) if x < 0 else 0)
    # df.ScanCount = df.ScanCount.map(lambda x: 0 if x < 0 else x)
    # return df


def transform_column(df, colname, keep_columns):
    dummied = pd.get_dummies(df[colname])
    return pd.concat([df[keep_columns], dummied], axis=1)


def transform_group(df, groupby_col, include_tripType=False):
    convert_dict = {x: np.sum for x in df.columns.tolist()}
    if include_tripType: convert_dict['TripType'] = np.mean
    return df.groupby(groupby_col).agg(convert_dict)


def convert_prediction(predictions, **kwargs):
    predictions = pd.Series(predictions)
    output_index = kwargs['output_index']
    actual_trip_types = kwargs['actual_trip_types']
    output = pd.get_dummies(predictions)
    output.columns = ["TripType_%i" % x for x in output.columns]
    missing_tt = ["TripType_%i" % x for x in list(
        set(actual_trip_types).difference(set(predictions.unique())))]
    print("Missing Trip Types:", missing_tt)
    for x in missing_tt:
        output[x] = 0
    return pd.concat([pd.Series(output_index), output], axis=1)
