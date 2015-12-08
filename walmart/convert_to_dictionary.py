from collections import Counter
import pandas as pd
import numpy as np
import argparse
import pickle


# then just access with
# y, X = zip(df_to_sentences(df))
def df_to_sentences(df):
    for q in df.VisitNumber.unique():
        cur_visit = df[df.VisitNumber == q].astype(str)
        words = []

        for col in set(df.columns).difference({"VisitNumber", "ScanCount"}):
            col_str = "_" + col[0]
            # just gets all the columns and converts them into a list
            # then adds that to words
            with_scan = cur_visit.ScanCount + "_" + cur_visit[col] + col_str
            no_scan = cur_visit[col] + col_str
            words += with_scan.as_matrix().flatten().tolist() + \
                     no_scan.as_matrix().flatten().tolist()
        yield (q, dict(Counter(words).most_common()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('filename')
    args = parser.parse_args()
    fname = args.filename

    df = pd.read_csv("data/" + fname + ".csv")
    with open(fname + ".pkl", "wb") as f:
        pickle.dump(list(df_to_sentences(df)), f)
