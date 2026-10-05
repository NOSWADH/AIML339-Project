import numpy as np
import tensorflow as tf

print("LOADED FROM:", __file__)

#walk-forward (expanding window) fine-tuning and evaluation, run
#separately per exercise since the combined sequence array is not one
#continuous timeline (build_sequences groups by exercise first).
#
#for each fold: trains a FRESH COPY of the pretrained model on every
#session chronologically before the test window, then evaluates on a
#fixed-size test window right after. Each fold starts from the ORIGINAL
#pretrained weights (not the previous fold's fine-tuned weights), so
#folds are independent -- standard practice for walk-forward evaluation.
def finetune_lstm_walkforward(pretrained_model,
                               hevy_sequences,
                               hevy_targets,
                               exercise_titles,
                               test_size=5,
                               step=None,
                               warmup_epochs=3,
                               epochs=10,
                               batch_size=8,
                               warmup_lr=0.001,
                               finetune_lr=0.0005,
                               min_train_size=8):
    if step is None:
        step = test_size

    original_weights = pretrained_model.get_weights()

    all_fold_results = []
    all_preds, all_targets, all_exercise_labels = [], [], []

    for exercise in np.unique(exercise_titles):
        idx = np.where(exercise_titles == exercise)[0]
        X_ex = hevy_sequences[idx]
        y_ex = hevy_targets[idx]
        n = len(X_ex)

        print(f"\n######## {exercise}: {n} sequences available ########")

        train_end = min_train_size
        fold_num = 0

        while train_end + test_size <= n:
            fold_num += 1
            test_start = train_end
            test_end = train_end + test_size

            X_train = X_ex[:train_end]
            y_train = y_ex[:train_end]
            X_test = X_ex[test_start:test_end]
            y_test = y_ex[test_start:test_end]

            print(f"  Fold {fold_num}: train on [0:{train_end}], "
                  f"test on [{test_start}:{test_end}]")

            #fresh clone, reset to the ORIGINAL pretrained weights --
            #never continues from a previous fold's fine-tuning
            model = tf.keras.models.clone_model(pretrained_model)
            model.set_weights(original_weights)

            #phase 1: warm up, only the final layer trains
            for layer in model.layers[:-1]:
                layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=warmup_lr),
                          loss='mse', metrics=['mae'])
            model.fit(X_train, y_train, epochs=warmup_epochs,
                     batch_size=batch_size, verbose=0)

            #phase 2: unfreeze everything, full fine-tune at a lower LR
            for layer in model.layers:
                layer.trainable = True
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
                          loss='mse', metrics=['mae'])
            model.fit(X_train, y_train, epochs=epochs,
                     batch_size=batch_size, verbose=0)

            preds = model.predict(X_test, verbose=0)

            all_fold_results.append({
                "exercise": exercise, "fold": fold_num,
                "train_size": train_end, "test_range": (test_start, test_end),
                "preds": preds, "targets": y_test
            })
            all_preds.append(preds)
            all_targets.append(y_test)
            all_exercise_labels.extend([exercise] * len(y_test))

            train_end += step

        print(f"  -> {fold_num} fold(s) completed for {exercise}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_exercise_labels = np.array(all_exercise_labels)

    print(f"\nTotal held-out predictions across all folds/exercises: {len(all_preds)}")

    return {
        "fold_results": all_fold_results,
        "all_preds": all_preds,
        "all_targets": all_targets,
        "exercise_labels": all_exercise_labels
    }