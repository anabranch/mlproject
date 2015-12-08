# first line: 32
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
