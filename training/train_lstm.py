import os
import numpy as np
from models.lstm import build_lstm
from utils.split import chronological_split
from utils.scaler import fit_scaler, transform_sequences
import tensorflow as tf

print("LOADED FROM:", __file__)

def train_lstm(kaggle_sequences, kaggle_targets, seq_len, feature_cols,
               save_path = "pretrained_lstm.keras",
               epochs = 20, batch_size = 32):

    #chronological split
    X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx = chronological_split(
    kaggle_sequences, kaggle_targets
    )


    #fit scaler on train only
    scaler = fit_scaler(X_train)

    #apply scaler everywhere
    X_train = transform_sequences(X_train, scaler)
    X_val = transform_sequences(X_val, scaler)
    X_test = transform_sequences(X_test, scaler)

    #build model
    num_features = len(feature_cols)
    model = build_lstm(seq_len=seq_len, num_features=num_features)

    #callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 5,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path,
        monitor = 'val_loss',
        save_best_only = True
    )

    #train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    #return everything needed for fine‑tuning
    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "history": history
    }

