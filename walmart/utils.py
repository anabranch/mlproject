import pandas as pd
import numpy as np


def convert_predictions(predictions):
    actual_trip_types = [999, 30, 26, 8, 35, 41, 21, 6, 42, 7, 9, 39, 25, 38,
                         15, 36, 20, 37, 32, 40, 5, 3, 4, 24, 33, 43, 31, 27,
                         34, 18, 29, 44, 19, 23, 22, 28, 14, 12]
    output_index = predictions['VisitNumber']
    output = pd.get_dummies(predictions['TripType'])

    missing_categories = set(actual_trip_types) \
        .difference(set(output.columns))
    print("Missing Categories", missing_categories)
    for missing in missing_categories:
        output[missing] = 0

    output = output[sorted(output.columns)]
    output.columns = ["TripType_%i" % x for x in output.columns]
    return pd.concat([output_index, output], axis=1).astype(int)
