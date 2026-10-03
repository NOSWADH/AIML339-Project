import numpy as np
from evaluation.metrics import mae, rmse, per_session_error

print("LOADED FROM:", __file__)

#exercise-specific filtering
def filter_exercise(X, y, exercise_titles, target_exercise):
    idx = np.where(exercise_titles == target_exercise)[0]
    return X[idx], y[idx]


#ML model predictions
def predict_lstm(model, X):
    return model.predict(X)

def predict_lr(lr_model, X_flat):
    return lr_model.predict(X_flat)

def predict_xgb(gb_weight, gb_reps, X_flat):
    w = gb_weight.predict(X_flat)
    r = gb_reps.predict(X_flat)
    return np.column_stack([w, r])


#non ML baseline predictions
def evaluate_nonml(baseline_dict, y_true):
    results = {}
    for name, pred in baseline_dict.items():
        results[name] = {
            "mae": mae(y_true[1:], pred),
            "rmse": rmse(y_true[1:], pred)
        }
    return results


#feature importance LR and GBT
def lr_feature_importance(lr_model, feature_names):
    return dict(zip(feature_names, lr_model.coef_[0]))

def xgb_feature_importance(gb_model, feature_names):
    scores = gb_model.feature_importances_
    return dict(zip(feature_names, scores))


#LSTM feature importance abolation
#train without one feature at a time and measure performance drop
def lstm_ablation(model_builder, X_train, y_train, X_val, y_val, feature_names):
    results = {}

    for i, feature in enumerate(feature_names):
        #remove feature i
        X_train_ab = np.delete(X_train, i, axis=2)
        X_val_ab   = np.delete(X_val,   i, axis=2)

        #build new model with fewer features
        model = model_builder(seq_len=X_train.shape[1], num_features=X_train_ab.shape[2])

        model.fit(X_train_ab, y_train, validation_data=(X_val_ab, y_val), epochs=5, verbose=0)

        preds = model.predict(X_val_ab)
        results[feature] = rmse(y_val, preds)

    return results


#function to evaluate everything at once
def evaluate_all(
    lstm_model,
    lr_model,
    gb_weight,
    gb_reps,
    baseline_nonml,
    X_test,
    y_test,
    X_test_flat,
    exercise_titles_test,
    feature_names,
    model_builder=None,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None
):
    results = {}

    #LSTM
    lstm_pred = predict_lstm(lstm_model, X_test)
    results["lstm"] = {"mae": mae(y_test, lstm_pred), "rmse": rmse(y_test, lstm_pred)}

    #LR
    lr_pred = predict_lr(lr_model, X_test_flat)
    results["lr"] = {"mae": mae(y_test, lr_pred),"rmse": rmse(y_test, lr_pred)}

    #GBT
    xgb_pred = predict_xgb(gb_weight, gb_reps, X_test_flat)
    results["xgb"] = {
        "mae": mae(y_test, xgb_pred),
        "rmse": rmse(y_test, xgb_pred)
    }

    #non ML baselines
    results["nonml"] = evaluate_nonml(baseline_nonml, y_test)

    #per exercise
    bench_idx = exercise_titles_test == "Bench Press (Barbell)"
    lat_idx   = exercise_titles_test == "Lat Pulldown (Cable)"

    results["bench"] = {
        "lstm_rmse": rmse(y_test[bench_idx], lstm_pred[bench_idx]),
        "lr_rmse": rmse(y_test[bench_idx], lr_pred[bench_idx]),
        "xgb_rmse": rmse(y_test[bench_idx], xgb_pred[bench_idx])
    }

    results["lat"] = {
        "lstm_rmse": rmse(y_test[lat_idx], lstm_pred[lat_idx]),
        "lr_rmse": rmse(y_test[lat_idx], lr_pred[lat_idx]),
        "xgb_rmse": rmse(y_test[lat_idx], xgb_pred[lat_idx])
    }

    #feature importance
    results["lr_feature_importance"] = lr_feature_importance(lr_model, feature_names)
    results["xgb_feature_importance_weight"] = xgb_feature_importance(gb_weight, feature_names)
    results["xgb_feature_importance_reps"] = xgb_feature_importance(gb_reps, feature_names)

    #LSTM ablation
    if model_builder is not None:
        results["lstm_ablation"] = lstm_ablation(
            model_builder,
            X_train, y_train,
            X_val, y_val,
            feature_names
        )

    return results
