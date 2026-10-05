import numpy as np

print("LOADED FROM:", __file__)

#splits sequences chronologically into train, validation, and test sets.
def chronological_split(X, y, val_ratio=0.2, test_ratio=0.15):
    n = len(X)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size

    train_idx = np.arange(0, train_size)
    val_idx   = np.arange(train_size, train_size + val_size)
    test_idx  = np.arange(train_size + val_size, n)

    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
        train_idx, val_idx, test_idx
    )
