import tensorflow as tf
 
print("LOADED FROM:", __file__)
 
#finetunes a pretrained LSTM model on the author's own Hevy data, using
#progressive unfreezing: first warm up the final layer only, then
#unfreeze everything for full fine-tuning at a lower learning rate.
#
#NOTE: trains on the FULL Hevy dataset, with no held-out validation or
#test split. The Hevy dataset is too small to split three ways without
#the splits becoming meaninglessly small. This is the function used to
#produce the final, deployable personalized model -- it is NOT used to
#evaluate personalized performance. Honest RMSE/MAE for the fine-tuned
#model is instead obtained separately via walk-forward evaluation
#(see finetune_lstm_walkforward), which trains and tests on several
#small rolling windows rather than one static split.
#
#hevy_sequences is expected to already be scaled (using the pretrain
#scaler) by the caller in main.ipynb.
def finetune_lstm(pretrained_model,
                  hevy_sequences,
                  hevy_targets,
                  seq_len,
                  feature_cols,
                  save_path="finetuned_lstm.keras",
                  warmup_epochs=3,
                  epochs=10,
                  batch_size=8,
                  warmup_lr=0.001,
                  finetune_lr=0.0005):
 
    X_hevy = hevy_sequences
    y_hevy = hevy_targets
 
    #--- phase 1: warm up, only the final layer trains ---
    for layer in pretrained_model.layers[:-1]:
        layer.trainable = False
 
    pretrained_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=warmup_lr),
        loss='mse',
        metrics=['mae']
    )
 
    pretrained_model.fit(
        X_hevy, y_hevy,
        epochs=warmup_epochs,
        batch_size=batch_size,
        verbose=1
    )
 
    #--- phase 2: unfreeze everything, full fine-tune at a lower LR ---
    for layer in pretrained_model.layers:
        layer.trainable = True
 
    pretrained_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
        loss='mse',
        metrics=['mae']
    )
 
    history = pretrained_model.fit(
        X_hevy, y_hevy,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
 
    pretrained_model.save(save_path)
 
    return {
        "model": pretrained_model,
        "history": history
    }