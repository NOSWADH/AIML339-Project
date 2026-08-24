from sklearn.linear_model import LinearRegression

print("LOADED FROM:", __file__)

#Trains a simple linear regression baseline.
#X_train shape: (samples, num_features)
#y_train shape: (samples, 2)
def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr
