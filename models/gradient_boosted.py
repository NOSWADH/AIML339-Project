from xgboost import XGBRegressor

print("LOADED FROM:", __file__)

#trains two XGBRegressor models
#one for next_weight and one for next_reps
def train_gradient_boost(X_train, y_train):
    gb_weight = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )

    gb_reps = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )

    gb_weight.fit(X_train, y_train[:, 0])
    gb_reps.fit(X_train, y_train[:, 1])

    return gb_weight, gb_reps
