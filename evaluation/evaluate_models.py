import numpy as np
from evaluation.metrics import mae, rmse

print("LOADED FROM:", __file__)


#ML model predictions -- LR/XGB were trained on SCALED targets, so their
#raw .predict() output is in scaled units and must be inverse-transformed
def predict_lr(lr_weight, lr_reps, X_flat_scaled, target_scaler):
    w = lr_weight.predict(X_flat_scaled)
    r = lr_reps.predict(X_flat_scaled)
    pred_scaled = np.column_stack([w, r])
    return target_scaler.inverse_transform(pred_scaled)


def predict_xgb(gb_weight, gb_reps, X_flat_scaled, target_scaler):
    w = gb_weight.predict(X_flat_scaled)
    r = gb_reps.predict(X_flat_scaled)
    pred_scaled = np.column_stack([w, r])
    return target_scaler.inverse_transform(pred_scaled)


def predict_lstm(model, X_scaled, target_scaler):
    pred_scaled = model.predict(X_scaled)
    return target_scaler.inverse_transform(pred_scaled)


#non ML baseline evaluation -- these predictions are ALREADY in raw
#units (see train_nonml_baselines, which operates on raw sequences), so
#no inverse-transform needed here. y_true passed in must also be raw.
def evaluate_nonml(baseline_dict, y_true_raw):
    results = {}
    for name, pred in baseline_dict.items():
        if pred.shape[0] != y_true_raw.shape[0]:
            raise ValueError(
                f"Baseline '{name}' has {pred.shape[0]} predictions but "
                f"y_true has {y_true_raw.shape[0]} rows -- these must match."
            )
        results[name] = {"mae": mae(y_true_raw, pred), "rmse": rmse(y_true_raw, pred)}
    return results


#feature importance -- computed in scaled-coefficient space, which is
#fine for RELATIVE importance ranking (the usual purpose of this
#analysis); inverse-transforming coefficients back to raw units is not
#meaningful for ranking purposes, so this is left as-is
def lr_feature_importance(lr_model, feature_names, seq_len):
    num_features = len(feature_names)
    coefs = lr_model.coef_.reshape(seq_len, num_features)
    importance = np.abs(coefs).sum(axis=0)
    return dict(zip(feature_names, importance))


def xgb_feature_importance(gb_model, feature_names, seq_len):
    num_features = len(feature_names)
    scores = gb_model.feature_importances_.reshape(seq_len, num_features)
    importance = scores.sum(axis=0)
    return dict(zip(feature_names, importance))


#LSTM feature importance via ablation -- operates entirely in scaled
#space (both training and the resulting rmse), consistent across all
#features being compared, so no inverse-transform needed for RELATIVE
#comparison between features
def lstm_ablation(model_builder, X_train, y_train, X_val, y_val, feature_names):
    results = {}
    for i, feature in enumerate(feature_names):
        X_train_ab = np.delete(X_train, i, axis=2)
        X_val_ab = np.delete(X_val, i, axis=2)

        model = model_builder(seq_len=X_train.shape[1], num_features=X_train_ab.shape[2])
        model.fit(X_train_ab, y_train, validation_data=(X_val_ab, y_val), epochs=5, verbose=0)

        preds = model.predict(X_val_ab)
        results[feature] = rmse(y_val, preds)
    return results


#evaluate everything at once, ALL IN RAW UNITS.
#
#X_test_scaled / X_test_flat_scaled: scaled features, fed to the models
#y_test_raw: the TRUE targets in raw units -- all error is computed
#  against this
#target_scaler: the SAME scaler used to scale y during training,
#  needed to inverse-transform LSTM/LR/XGB predictions back to raw units
def evaluate_all(
    lstm_model,
    lr_weight, lr_reps,
    gb_weight, gb_reps,
    baseline_nonml,
    X_test_scaled,
    y_test_raw,
    X_test_flat_scaled,
    target_scaler,
    exercise_titles_test,
    feature_names,
    seq_len,
    model_builder=None,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None
):
    results = {}

    #LSTM -- predict in scaled space, inverse-transform to raw
    lstm_pred = predict_lstm(lstm_model, X_test_scaled, target_scaler)
    results["lstm"] = {"mae": mae(y_test_raw, lstm_pred), "rmse": rmse(y_test_raw, lstm_pred)}

    #LR
    lr_pred = predict_lr(lr_weight, lr_reps, X_test_flat_scaled, target_scaler)
    results["lr"] = {"mae": mae(y_test_raw, lr_pred), "rmse": rmse(y_test_raw, lr_pred)}

    #GBT
    xgb_pred = predict_xgb(gb_weight, gb_reps, X_test_flat_scaled, target_scaler)
    results["xgb"] = {"mae": mae(y_test_raw, xgb_pred), "rmse": rmse(y_test_raw, xgb_pred)}

    #non ML baselines -- already raw, compared directly against y_test_raw
    results["nonml"] = evaluate_nonml(baseline_nonml, y_test_raw)

    #per-exercise breakdown (all in raw units now)
    def safe_rmse(a, b):
        return rmse(a, b) if len(a) > 0 else None

    if exercise_titles_test is not None:
        bench_idx = exercise_titles_test == "Bench Press"
        lat_idx = exercise_titles_test == "Lat Pulldown"

        results["bench"] = {
            "lstm_rmse": safe_rmse(y_test_raw[bench_idx], lstm_pred[bench_idx]),
            "lr_rmse": safe_rmse(y_test_raw[bench_idx], lr_pred[bench_idx]),
            "xgb_rmse": safe_rmse(y_test_raw[bench_idx], xgb_pred[bench_idx]),
        }
        results["lat"] = {
            "lstm_rmse": safe_rmse(y_test_raw[lat_idx], lstm_pred[lat_idx]),
            "lr_rmse": safe_rmse(y_test_raw[lat_idx], lr_pred[lat_idx]),
            "xgb_rmse": safe_rmse(y_test_raw[lat_idx], xgb_pred[lat_idx]),
        }

    #feature importance (relative ranking, computed in scaled space)
    results["lr_feature_importance_weight"] = lr_feature_importance(lr_weight, feature_names, seq_len)
    results["lr_feature_importance_reps"] = lr_feature_importance(lr_reps, feature_names, seq_len)
    results["xgb_feature_importance_weight"] = xgb_feature_importance(gb_weight, feature_names, seq_len)
    results["xgb_feature_importance_reps"] = xgb_feature_importance(gb_reps, feature_names, seq_len)

    #LSTM ablation (optional)
    if model_builder is not None:
        results["lstm_ablation"] = lstm_ablation(
            model_builder, X_train, y_train, X_val, y_val, feature_names
        )

    return results


def evaluate_lstm_only(
    lstm_model,
    X_test_scaled,
    y_test_raw,
    target_scaler,
    exercise_titles_test,
    feature_names,
    seq_len
):
    lstm_pred = target_scaler.inverse_transform(
        lstm_model.predict(X_test_scaled)
    )

    return {
        "lstm": {
            "mae": mae(y_test_raw, lstm_pred),
            "rmse": rmse(y_test_raw, lstm_pred)
        }
    }

def evaluate_baselines_only(
    lr_weight, lr_reps,
    gb_weight, gb_reps,
    baseline_nonml,
    X_test_flat_scaled,
    y_test_raw,
    target_scaler
):
    results = {}

    # LR
    lr_pred = predict_lr(lr_weight, lr_reps, X_test_flat_scaled, target_scaler)
    results["lr"] = {
        "mae": mae(y_test_raw, lr_pred),
        "rmse": rmse(y_test_raw, lr_pred)
    }

    # XGB
    xgb_pred = predict_xgb(gb_weight, gb_reps, X_test_flat_scaled, target_scaler)
    results["xgb"] = {
        "mae": mae(y_test_raw, xgb_pred),
        "rmse": rmse(y_test_raw, xgb_pred)
    }

    # Non‑ML baselines (already raw)
    results["nonml"] = evaluate_nonml(baseline_nonml, y_test_raw)

    return results
