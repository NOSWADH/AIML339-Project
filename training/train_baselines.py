import numpy as np
from models.linear_regression import train_linear_regression
from models.gradient_boosted import train_gradient_boost
from utils.split import chronological_split
from utils.scaler import fit_scaler, transform_sequences

print("LOADED FROM:", __file__)

#flattens LSTM sequences into tabular format for baseline models.
def flatten_sequences(X):
    samples, seq_len, num_features = X.shape
    return X.reshape(samples, seq_len * num_features)



#trains baseline models (Linear Regression + XGBoost)using flattened Kaggle sequences.
def train_baselines(kaggle_sequences, kaggle_targets, seq_len, feature_cols):
    
    #chronological split
    X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx = chronological_split(
    kaggle_sequences, kaggle_targets
    )


    #fit scaler on TRAIN only (same as LSTM)
    scaler = fit_scaler(X_train)

    #apply scaler everywhere
    X_train = transform_sequences(X_train, scaler)
    X_val = transform_sequences(X_val, scaler)
    X_test = transform_sequences(X_test, scaler)

    #flatten sequences for baseline models
    X_train_flat = flatten_sequences(X_train)
    X_val_flat = flatten_sequences(X_val)
    X_test_flat = flatten_sequences(X_test)

    #train linear regression
    lr_model = train_linear_regression(X_train_flat, y_train)

    #train gradient boost
    gb_weight, gb_reps = train_gradient_boost(X_train_flat, y_train)

    #package results
    return {
        "lr_model": lr_model,
        "gb_weight": gb_weight,
        "gb_reps": gb_reps,
        "X_val": X_val_flat,
        "y_val": y_val,
        "X_test": X_test_flat,
        "y_test": y_test,
        "scaler": scaler,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx
    }


#non ML baselines

#predicts next is same as last
def baseline_recent_performance(y_true):
    return y_true[:-1]  #shift forward by 1



#add 1 rep per session
def baseline_standard_progression_plus_rep(sequences, targets):
    
    #extract last weight/rep from each sequence
    last_weights = sequences[:, -1, 0]
    last_reps = sequences[:, -1, 1]

    pred_weights = last_weights[:-1]
    pred_reps = last_reps[:-1] + 1

    return np.column_stack([pred_weights, pred_reps])



#add 2.5kg per session
def baseline_standard_progression_plus_weight(sequences, targets):
    
    #extract last weight/rep from each sequence
    last_weights = sequences[:, -1, 0]
    last_reps = sequences[:, -1, 1]

    pred_weights = last_weights[:-1] + 2.5
    pred_reps = last_reps[:-1]

    return np.column_stack([pred_weights, pred_reps])


#epley 1RM formula
def baseline_epley(sequences):
    last_weights = sequences[:, -1, 0]
    last_reps = sequences[:, -1, 1]

    #epley 1RM
    one_rm = last_weights * (1 + last_reps / 30.0)

    pred_weights = one_rm[:-1]
    pred_reps = np.ones_like(pred_weights)

    return np.column_stack([pred_weights, pred_reps])


#returns predictions for all non-ML baselines.
def train_nonml_baselines(sequences, targets):

    last_val_pred = baseline_recent_performance(targets)
    standard_weight_prog_pred = baseline_standard_progression_plus_weight(sequences, targets)
    standard_reps_prog_pred = baseline_standard_progression_plus_rep(sequences, targets)
    epley_pred = baseline_epley(sequences)

    return {
        "last_value": last_val_pred,
        "standard_weight_progression": standard_weight_prog_pred,
        "standard_reps_progression": standard_reps_prog_pred,
        "epley": epley_pred
    }
