import pandas as pd
import numpy as np


def transform_column(df, colname, keep_columns):
    dummied = pd.get_dummies(df[colname])
    return pd.concat([df[keep_columns], dummied], axis=1)


def transform_group(df, groupby_col):
    convert_dict = {x: np.sum for x in df.columns.tolist()}
    convert_dict['TripType'] = np.mean  # need to abstract
    return df.groupby(groupby_col).agg(convert_dict)
