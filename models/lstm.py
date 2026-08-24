import tensorflow as tf
from tensorflow.keras import layers, models

print("LOADED FROM:", __file__)

#builds a simple LSTM model for predicting next_weight and next_reps.
def build_lstm(seq_len, num_features, hidden_units=64):
    model = models.Sequential([
        layers.Input(shape=(seq_len, num_features)),
        layers.LSTM(hidden_units, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)  #next_weight, next_reps
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='rmse', metrics=['mae'])

    return model
