import numpy as np
from sklearn.preprocessing import StandardScaler

print("LOADED FROM:", __file__)

#fits a StandardScaler on the flattened training sequences.
def fit_scaler(X_train):
    num_features = X_train.shape[2]
    X_flat = X_train.reshape(-1, num_features)

    scaler = StandardScaler()
    scaler.fit(X_flat)
    return scaler

#applies the fitted scaler to any sequence set (train/val/test/hevy).
def transform_sequences(X, scaler):
    num_features = X.shape[2]
    X_flat = X.reshape(-1, num_features)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(X.shape)
