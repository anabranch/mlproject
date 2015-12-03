from collections import Counter
import pandas as pd
import numpy as np
import pickle


# then just access with
# y, X = zip(df_to_sentences(df))
def df_to_sentences(df):
    for q in df.VisitNumber.unique():
        words = []
        for col in set(df.columns).difference({"VisitNumber"}):
            col_str = "_" + col[0]
            # just gets all the columns and converts them into a list
            # then adds that to words
            words += list(map(lambda x: x + col_str,
                              df[df.VisitNumber == q][col].as_matrix().flatten(
                              ).astype(str).tolist()))
        yield (q, dict(Counter(words).most_common()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('filename')
    args = parser.parse_args()
    fname = args.filename

    df = pd.read_csv("data/" + fname + ".csv")
    with open(fname + ".pkl", "wb") as f:
        pickle.dump(list(df_to_sentences(df)), f)
