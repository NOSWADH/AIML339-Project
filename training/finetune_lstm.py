import tensorflow as tf
from utils.scaler import transform_sequences

print("LOADED FROM:", __file__)

#finetunes a pretrained LSTM model on hevy data.
def finetune_lstm(pretrained_model,
                  scaler,
                  hevy_sequences,
                  hevy_targets,
                  seq_len,
                  feature_cols,
                  save_path = "finetuned_lstm.keras",
                  epochs = 10,
                  batch_size = 32,
                  learning_rate = 0.0005):

    #scale hevy sequences using kaggle's scaler
    X_hevy = transform_sequences(hevy_sequences, scaler)
    y_hevy = hevy_targets

    #lower learning rate for finetuning
    pretrained_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = 'mse',
        metrics = ['mae']
    )

    #finetune
    history = pretrained_model.fit(
        X_hevy, y_hevy,
        epochs = epochs,
        batch_size = batch_size,
        verbose = 1
    )

    #save personalized model
    pretrained_model.save(save_path)

    return {
        "model": pretrained_model,
        "history": history
    }
