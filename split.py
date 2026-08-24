import numpy as np

print("LOADED FROM:", __file__)

#splits sequences chronologically into train, validation, and test sets.
def chronological_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = X.shape[0]

    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test