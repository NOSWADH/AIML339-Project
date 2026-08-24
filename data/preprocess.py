#narrow down to select exercises
#sort by date and exercise
#engineer proposed features
#add target features

import pandas as pd
import numpy as np

print("LOADED FROM:", __file__)

def preprocess(df, exercises):
    df = df[df['exercise_title'].isin(exercises)].copy()
    df = df.sort_values(['exercise_title', 'date'])
    
    #weight related features

    #session volume
    df['session_volume'] = (df['weight_kg'] * df['reps']).groupby(df['date']).transform('sum')

    #just take the topset for each exercise
    df['max_weight_for_day'] = df.groupby(['date', 'exercise_title'])['weight_kg'].transform('max')
    df = df[df['weight_kg'] == df['max_weight_for_day']]
    df = df.sort_values(['date', 'exercise_title', 'reps'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['date', 'exercise_title'], keep='first')
    df = df.drop(columns=['max_weight_for_day'])

    #previous top set
    df['prev_top_weight'] = df.groupby('exercise_title')['weight_kg'].shift(1)
    df['prev_top_reps']   = df.groupby('exercise_title')['reps'].shift(1)
    df['prev_top_weight'] = df['prev_top_weight'].fillna(df['weight_kg'])
    df['prev_top_reps'] = df['prev_top_reps'].fillna(df['reps'])

    #date related features
    
    #time of day
    def time_of_day(hour):
        if 6 <= hour < 12:
            return 0 #morning
        elif 12 <= hour < 18:
            return 1 #afternoon
        elif 18 <= hour < 24:
            return 2 #evening
        else:
            return 3 #night

    df['hour'] = df['date'].dt.hour
    df['time_of_day'] = df['hour'].apply(time_of_day)
    df.drop(columns='hour', inplace=True)

    #days since last performed
    df['prev_date'] = df.groupby('exercise_title')['date'].shift(1)
    df['days_since_last'] = (df['date'] - df['prev_date']).dt.days
    df = df.drop(columns=['prev_date'])
    df['days_since_last'] = df.groupby('exercise_title')['days_since_last'].transform(lambda s: s.fillna(s.median()))

    #progression trend
    #calculate slope of last x sessions
    def rolling_slope(x):
        if len(x) < 2:
            return np.nan
        y = x.values
        t = np.arange(len(y))
        slope = np.polyfit(t, y, 1)[0]
        return slope

    #slope over last 5 sessions
    df['trend_slope'] = (df.groupby('exercise_title')['weight_kg'].rolling(5, min_periods=2).apply(rolling_slope).reset_index(level=0, drop=True))
    df['trend_slope'] = df['trend_slope'].round(2)
    df['trend_slope'] = df['trend_slope'].fillna(0)

    #I wanted to do a feature on days since last low volume day but with the data available the feature had little significance so I scrapped it.

    #target features
    
    df = df.sort_values(['exercise_title', 'date'])
    df['next_weight'] = df.groupby('exercise_title')['weight_kg'].shift(-1)
    df['next_reps']   = df.groupby('exercise_title')['reps'].shift(-1)
    df = df.dropna(subset=['next_weight', 'next_reps'])

    return df