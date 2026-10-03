import numpy as np

print("LOADED FROM:", __file__)

#mean absolute error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

#root mean squared error
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

#error per session
def per_session_error(y_true, y_pred):
    mae_per = np.abs(y_true - y_pred)
    rmse_per = (y_true - y_pred)**2
    return mae_per, rmse_per
