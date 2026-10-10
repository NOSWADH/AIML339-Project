import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from utils.split import chronological_split

print("LOADED FROM:", __file__)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def flatten_sequences(X):
    """(samples, seq_len, num_features) -> (samples, seq_len * num_features)"""
    samples, seq_len, num_features = X.shape
    return X.reshape(samples, seq_len * num_features)


# ---------------------------------------------------------
# Non-ML baselines
#
# These are FORMULAS defined in real units (kg, reps) -- "+2.5kg" and
# the Epley 1RM formula only make physical sense on raw, unscaled
# values. They must always receive the RAW (unscaled) sequences, never
# the scaled ones, regardless of how the ML models are trained.
# ---------------------------------------------------------

def baseline_last_value(sequences_raw):
    last_weights = sequences_raw[:, -1, 0]
    last_reps = sequences_raw[:, -1, 1]
    return np.column_stack([last_weights, last_reps])


def baseline_standard_weight_progression(sequences_raw):
    last_weights = sequences_raw[:, -1, 0]
    last_reps = sequences_raw[:, -1, 1]
    return np.column_stack([last_weights + 2.5, last_reps])


def baseline_standard_reps_progression(sequences_raw):
    last_weights = sequences_raw[:, -1, 0]
    last_reps = sequences_raw[:, -1, 1]
    return np.column_stack([last_weights, last_reps + 1])


def baseline_epley(sequences_raw):
    last_weights = sequences_raw[:, -1, 0]
    last_reps = sequences_raw[:, -1, 1]
    one_rm = last_weights * (1 + last_reps / 30.0)
    return np.column_stack([one_rm, np.ones_like(one_rm)])


def train_nonml_baselines(sequences_raw):
    """Returns PRECOMPUTED prediction arrays, in RAW units, each shape
    (n_samples, 2), aligned 1:1 with whatever raw targets correspond to
    `sequences_raw`.
    """
    return {
        "last_value": baseline_last_value(sequences_raw),
        "standard_weight_progression": baseline_standard_weight_progression(sequences_raw),
        "standard_reps_progression": baseline_standard_reps_progression(sequences_raw),
        "epley": baseline_epley(sequences_raw),
    }


# ---------------------------------------------------------
# ML baselines (Ridge regression + XGBoost), trained on SCALED
# features/targets (consistent with how the LSTM is trained). Their
# predictions therefore come out in SCALED units too, and must be
# inverse-transformed back to raw units at evaluation time -- see
# evaluate_models.py.
# ---------------------------------------------------------

def train_ridge_baseline(X_train_flat_scaled, y_train_scaled):
    lr_weight = Ridge(alpha=1.0)
    lr_reps = Ridge(alpha=1.0)
    lr_weight.fit(X_train_flat_scaled, y_train_scaled[:, 0])
    lr_reps.fit(X_train_flat_scaled, y_train_scaled[:, 1])
    return {"lr_weight": lr_weight, "lr_reps": lr_reps}


def train_xgb_baseline(X_train_flat_scaled, y_train_scaled):
    xgb_kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror",
    )
    gb_weight = XGBRegressor(**xgb_kwargs)
    gb_reps = XGBRegressor(**xgb_kwargs)
    gb_weight.fit(X_train_flat_scaled, y_train_scaled[:, 0])
    gb_reps.fit(X_train_flat_scaled, y_train_scaled[:, 1])
    return {"gb_weight": gb_weight, "gb_reps": gb_reps}


# ---------------------------------------------------------
# Full baseline trainer.
#
# Takes BOTH the scaled sequences/targets (used to fit the ML models,
# and to produce the chronological split) AND the raw, unscaled
# sequences/targets (used for the non-ML baselines and for reporting
# the true test targets in real units). The split is computed once, on
# the scaled data, and the resulting indices are reused to slice the
# raw arrays -- guaranteeing both views refer to exactly the same
# sessions.
# ---------------------------------------------------------

def train_baselines(kaggle_sequences_scaled, kaggle_targets_scaled,
                     kaggle_sequences_raw, kaggle_targets_raw,
                     seq_len, feature_cols):

    #split the SCALED data -- this defines the train/val/test boundary
    X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, train_idx, val_idx, test_idx = chronological_split(
        kaggle_sequences_scaled, kaggle_targets_scaled
    )

    #reuse the SAME indices to slice the RAW arrays, so both views are
    #guaranteed to refer to the exact same sessions
    X_test_raw = kaggle_sequences_raw[test_idx]
    y_test_raw = kaggle_targets_raw[test_idx]

    X_train_flat_scaled = flatten_sequences(X_train_s)
    X_test_flat_scaled = flatten_sequences(X_test_s)

    #fit ML baselines on SCALED train data only
    lr_models = train_lr_baseline(X_train_flat_scaled, y_train_s)
    xgb_models = train_xgb_baseline(X_train_flat_scaled, y_train_s)

    #non-ML baselines computed directly on RAW test sequences
    baseline_nonml = train_nonml_baselines(X_test_raw)

    return {
        "lr_weight": lr_models["lr_weight"],
        "lr_reps": lr_models["lr_reps"],
        "gb_weight": xgb_models["gb_weight"],
        "gb_reps": xgb_models["gb_reps"],
        "baseline_nonml": baseline_nonml,
        "X_test_flat_scaled": X_test_flat_scaled,
        "y_test_scaled": y_test_s,
        "y_test_raw": y_test_raw,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }