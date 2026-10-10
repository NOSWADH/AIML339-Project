# Predicting Next Session Lifting Performance with a Personalised LSTM
## AIML339 project, Dawson Howarth (300661017), Victoria University of Wellington

This project predicts a lifter's next session top set weight and reps for bench press and lat pulldown from their Hevy workout log. An LSTM is pretrained on a public single lifter log from Kaggle and then finetuned on my own log using two phase progressive unfreezing. It is compared against progression heuristics (persistence, +2.5 kg, +1 rep, Epley 1RM), ridge regression and XGBoost. Evaluation uses an expanding window walk forward protocol repeated over multiple random seeds, with paired Wilcoxon tests.

### Main result (20 seeds, 85 walk forward test windows)
The finetuned LSTM has MAE 9.91 +- 0.53 and RMSE 16.46 +- 0.59.
Persistence has MAE 8.19 and RMSE 17.08.
Finetuning clearly beats every model that never sees personal data.
It does not beat simply repeating the last session: no difference against the heuristics is significant after Holm correction.

## Repository Structure
AIML339/
    main.ipynb                  # Runs the whole pipeline: data, baselines, walk forward, stats, plots
        data/
            load_data.py            # Load Kaggle / Hevy CSVs, drop unused columns, lb -> kg, parse dates
            preprocess.py           # Exercise name normalisation, top set extraction, feature engineering, targets
            sequence_builder.py     # 5 session sliding windows per exercise, (X, y, exercise labels)
            Hevy_workouts_log_100_weeks.csv   # Kaggle pretraining log (see "Data")
            new_workout_data.csv              # My personal Hevy export (see "Data")
        models/
            lstm.py                 # LSTM(64) -> Dense(32, ReLU) -> Dense(2)
        training/
            train_lstm.py           # Pretraining on Kaggle (chronological split, early stopping)
            finetune_walk.py        # Walk forward finetuning + all baselines + ablation (MAIN EVALUATION)
            finetune_lstm.py        # Finetunes on ALL personal data -> final deployable model (not used for evaluation)
            train_baselines.py      # Heuristic baselines, ridge and XGBoost on Kaggle data
        evaluation/
            metrics.py              # MAE, RMSE
            evaluate_models.py      # Kaggle test split evaluation of the baselines
        utils/
            scaler.py               # StandardScaler fitting/applying on 3D sequence arrays
            split.py                # Chronological train/val/test split (65/20/15)

models/linear_regression.py and models/gradient_boosted.py are early versions of the baselines and are not used. The baselines actually used are defined in training/train_baselines.py (ridge, a = 1, and XGBoost) and training/finetune_walk.py (ridge refitted per fold).

## Requirements
Python 3.10+
TensorFlow 2.x (CPU is fine; native Windows has no GPU support for TF >= 2.11)
scikit-learn, XGBoost, NumPy, pandas, SciPy, Matplotlib, Jupyter
bash
pip install tensorflow scikit-learn xgboost numpy pandas scipy matplotlib jupyter

The results in the report were produced on Windows 11 with an Intel Core CPU (no GPU).

## Data
Both datasets use the Hevy export schema (title, start_time, end_time, exercise_title, set_index, set_type, weight_*, reps, ...).

File	Source	Used for
data/Hevy_workouts_log_100_weeks.csv	Kaggle: Hevy App Workout Dataset	Pretraining (190 sessions -> 182 windows)
data/new_workout_data.csv	My own Hevy export (Aug 2025 - Oct 2026)	Finetuning and walk forward evaluation (117 sessions -> 109 windows)

### Preprocessing summary (data/preprocess.py)
Exercise names are normalised by keyword. Note that this merges incline, dumbbell and Smith machine variants into "Bench Press", and cable, machine and high row variants into "Lat Pulldown" (discussed as a limitation in the report).
Each session is reduced to its top set: the heaviest set, with ties broken by more reps.
Seven features are computed per session: weight_kg, reps, session_volume, prev_top_weight, prev_top_reps, days_since_last, trend_slope.
The targets are the next session's next_weight and next_reps for the same exercise.
Windows of 5 consecutive sessions (seq_len = 5) form the model input. Windows never cross exercises.

## How To Run
Open main.ipynb and run the cells in order:

Cell	What it does	Output
0	Imports
2	Load + preprocess both logs, build windows	Prints session/window counts
3	Fit feature/target scalers
4	Train + evaluate baselines on the Kaggle test split	Report Table I
5	Multi seed experiment: pretrain on Kaggle, then walk forward finetuning on my data with all baselines scored on the same windows	walkforward_predictions.csv, walkforward_ablation.csv, pretrain_histories.json; prints parameter count and timings
6	Summary table (mean +- SD over seeds), overall and per exercise	Report Table II, walkforward_summary*.csv
7	Paired Wilcoxon signed rank tests, Holm correction, rank biserial effect size	Report Table III, walkforward_tests.csv
8	Ablation importance summary	walkforward_ablation_summary.csv
Plots	Reads the saved CSV/JSON files and draws every report figure	fig1_png - fig6_png

Cells 6-8 and the plotting cell only read the saved files, so they can be rerun without repeating the experiment.

Runtime: about 2.6 min per seed on a laptop CPU (pretraining = 9 s, plus 17 folds x = 9 s finetuning), so the 20 seed run takes about 1 hour. Set SEEDS = [1, 2] for a quick check first.

## Configuration used for the report
Setting	Value
Seeds	SEEDS = list(range(1, 21)) (seeds 1-20); each sets Python, NumPy and TensorFlow
Ablation seeds	all 20 (ABLATION_SEEDS = 20)
Kaggle split	chronological 65/20/15 -> 119 / 36 / 27 windows
LSTM	LSTM(64) -> Dense(32, ReLU) -> Dense(2); 20,578 parameters; Adam; MSE loss
Pretraining	batch 32, lr 1e-3, <= 20 epochs, early stopping (patience 5, best weights restored)
Finetuning phase 1	only the output layer trainable, 3 epochs, lr 1e-3, batch 8
Finetuning phase 2	all layers trainable, 20 epochs, lr 5e-4, batch 8
Walk forward	per exercise; first fold trains on 8 windows; test block 5; step 5 -> 9 bench + 8 lat folds = 85 test windows; each fold starts from a fresh copy of the pretrained weights
Ridge	a = 1.0 (Kaggle only and refitted per fold on personal data)
XGBoost	300 trees, max depth 4, lr 0.05, subsample 0.8, colsample 0.8, lambda = 1
Statistics	two sided Wilcoxon signed rank on per window MAE (LSTM averaged over seeds), Holm adjusted over 8 comparisons, a = 0.05

Metrics: MAE and RMSE in original units, averaged over both targets (kg and reps). Per target MAE (mae_w, mae_r) is also reported.

## Notes and known limitations
Exercise variant mixing creates artificial load jumps, for example 85 -> 34 -> 85 kg on bench. This inflates every model's weight error.
The scalers are fitted on pooled Kaggle and personal data, including later personal sessions (a mild leak).
There is no per fold validation set, so the number of finetuning epochs is fixed rather than selected.
session_volume sums both lifts in a workout and includes warmup sets.
An earlier fixed five session hold out evaluation was discarded because finetune_lstm.py trains on all personal data, including those sessions. An off by one error in sequence_builder.py, which made the target two sessions ahead, was also fixed. All reported results come from the corrected walk forward pipeline.