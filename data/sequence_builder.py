#build the sequences used for the LSTM model
 
import numpy as np
 
print("LOADED FROM:", __file__)
 
def build_sequences(df, feature_cols, target_cols, seq_len=5):
    df = df.sort_values(['exercise_title', 'date'])
    X_list, y_list, ex_list = [], [], []
 
    for ex, group in df.groupby('exercise_title'):
        X_vals = group[feature_cols].values
        y_vals = group[target_cols].values
 
        for i in range(len(group) - seq_len + 1):
            X_list.append(X_vals[i:i+seq_len])
            y_list.append(y_vals[i+seq_len-1])  #next session after the window
            ex_list.append(ex)
 
    X = np.array(X_list)        # (samples, seq_len, num_features)
    y = np.array(y_list)        # (samples, 2)
    ex = np.array(ex_list)      # (samples,)
    return X, y, ex