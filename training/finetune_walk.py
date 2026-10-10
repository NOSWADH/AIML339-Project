import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import Ridge

print("LOADED FROM:", __file__)

#walk-forward (expanding window) fine-tuning and evaluation, run
#separately per exercise.
#
#for each fold: a FRESH copy of the pretrained model is fine-tuned on
#every window chronologically before the test block, then predicts the
#next `test_size` windows. Every baseline is scored on exactly the same
#test windows, so all results are paired and directly comparable.
#
#CHANGED vs previous version:
# - predictions are inverse-transformed to raw kg/reps before being stored
# - baselines (persistence, heuristics, Epley, personal Ridge refit per
#   fold, Kaggle-only Ridge/XGB, pretrained-only LSTM) are scored on the
#   same windows
# - LSTM ablation is done per fold on the held-out windows
# - fit time per fold is recorded
# - epochs default raised to 20 to match finetune_lstm
# - returns tidy DataFrames instead of raw arrays

def _flat(X):
    return X.reshape(len(X), -1)


#calling the model directly is much faster than model.predict() for tiny
#batches: predict() builds a new tf.function for every freshly cloned
#model, which is slow (especially on Windows) when called hundreds of times
def _pred(model, X):
    return model(X, training=False).numpy()


def _row_records(store, exercise, fold, win_idx, model_name, y_true, y_pred):
    for j in range(len(y_true)):
        store.append({
            "exercise": exercise, "fold": fold, "window": int(win_idx[j]),
            "model": model_name,
            "true_w": y_true[j, 0], "true_r": y_true[j, 1],
            "pred_w": y_pred[j, 0], "pred_r": y_pred[j, 1],
        })


def finetune_lstm_walkforward(pretrained_model,
                              X_scaled, y_scaled,      #scaled windows/targets (model inputs)
                              X_raw, y_raw,            #raw windows/targets (heuristics + scoring)
                              exercise_titles,         #from build_sequences
                              target_scaler,
                              feature_cols,
                              kaggle_models=None,      #baseline_result dict from train_baselines
                              test_size=5,
                              step=None,
                              min_train_size=8,
                              warmup_epochs=3,
                              epochs=20,
                              batch_size=8,
                              warmup_lr=0.001,
                              finetune_lr=0.0005,
                              do_ablation=True):
    if step is None:
        step = test_size

    original_weights = pretrained_model.get_weights()
    inv = target_scaler.inverse_transform

    pred_rows, ablation_rows, fit_times = [], [], []

    for exercise in np.unique(exercise_titles):
        idx = np.where(exercise_titles == exercise)[0]
        Xs, ys, Xr, yr = X_scaled[idx], y_scaled[idx], X_raw[idx], y_raw[idx]
        n = len(Xs)

        train_end, fold = min_train_size, 0
        while train_end + test_size <= n:
            fold += 1
            tr = slice(0, train_end)
            te = slice(train_end, train_end + test_size)
            win_idx = idx[te]
            y_true = yr[te]

            #--- fine-tuned LSTM (fresh copy of the pretrained weights) ---
            model = tf.keras.models.clone_model(pretrained_model)
            model.set_weights(original_weights)

            t0 = time.perf_counter()
            for layer in model.layers[:-1]:
                layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=warmup_lr), loss='mse')
            model.fit(Xs[tr], ys[tr], epochs=warmup_epochs, batch_size=batch_size, verbose=0)

            for layer in model.layers:
                layer.trainable = True
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr), loss='mse')
            model.fit(Xs[tr], ys[tr], epochs=epochs, batch_size=batch_size, verbose=0)
            fit_times.append(time.perf_counter() - t0)

            lstm_pred = inv(_pred(model, Xs[te]))
            _row_records(pred_rows, exercise, fold, win_idx, "lstm_finetuned", y_true, lstm_pred)

            #--- pretrained-only LSTM (no personal data) ---
            _row_records(pred_rows, exercise, fold, win_idx, "lstm_pretrained_only", y_true,
                         inv(_pred(pretrained_model, Xs[te])))

            #--- heuristics on RAW windows (last session = current session) ---
            last = Xr[te][:, -1, :2]   #weight_kg, reps of the latest session
            _row_records(pred_rows, exercise, fold, win_idx, "persistence", y_true, last)
            _row_records(pred_rows, exercise, fold, win_idx, "plus_2.5kg", y_true, last + np.array([2.5, 0]))
            _row_records(pred_rows, exercise, fold, win_idx, "plus_1rep", y_true, last + np.array([0, 1]))
            epley = np.column_stack([last[:, 0] * (1 + last[:, 1] / 30.0), np.ones(len(last))])
            _row_records(pred_rows, exercise, fold, win_idx, "epley", y_true, epley)

            #--- personal Ridge, refit on the same training windows as the LSTM ---
            ridge = Ridge(alpha=1.0).fit(_flat(Xs[tr]), ys[tr])
            _row_records(pred_rows, exercise, fold, win_idx, "ridge_personal", y_true,
                         inv(ridge.predict(_flat(Xs[te]))))

            #--- Kaggle-only Ridge / XGB (no personal data) ---
            if kaggle_models is not None:
                Xf = _flat(Xs[te])
                lr = np.column_stack([kaggle_models["lr_weight"].predict(Xf),
                                      kaggle_models["lr_reps"].predict(Xf)])
                gb = np.column_stack([kaggle_models["gb_weight"].predict(Xf),
                                      kaggle_models["gb_reps"].predict(Xf)])
                _row_records(pred_rows, exercise, fold, win_idx, "ridge_kaggle", y_true, inv(lr))
                _row_records(pred_rows, exercise, fold, win_idx, "xgb_kaggle", y_true, inv(gb))

            #--- ablation: zero one (scaled) feature at test time = replace with mean ---
            for f, fname in enumerate(feature_cols if do_ablation else []):
                X_ab = Xs[te].copy()
                X_ab[:, :, f] = 0.0
                p = inv(_pred(model, X_ab))
                for j in range(len(y_true)):
                    ablation_rows.append({
                        "exercise": exercise, "fold": fold, "window": int(win_idx[j]),
                        "feature": fname,
                        "true_w": y_true[j, 0], "true_r": y_true[j, 1],
                        "pred_w": p[j, 0], "pred_r": p[j, 1],
                    })

            train_end += step

        print(f"{exercise}: {n} windows, {fold} folds, {fold * test_size} test predictions")

    return pd.DataFrame(pred_rows), pd.DataFrame(ablation_rows), fit_times