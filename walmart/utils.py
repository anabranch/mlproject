import pandas as pd
import numpy as np


def convert_predictions(predictions, **kwargs):
    predictions = pd.Series(predictions)
    output_index = kwargs['output_index']
    actual_trip_types = kwargs['actual_trip_types']
    output = pd.get_dummies(predictions)
    missing_categories = set(actual_trip_types).difference(set(output.columns))
    print("Missing Categories", missing_categories)
    for missing in missing_categories:
        output[missing] = 0
    output = output[sorted(output.columns)]
    output.columns = ["TripType_%i" % x for x in output.columns]
    return pd.concat([pd.Series(output_index), output], axis=1).astype(int)
