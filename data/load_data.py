#Load the data into a dataframe for each dataset
#prune unneeded features
#convert measurements
#change start_time to use date format

import pandas as pd

print("LOADED FROM:", __file__)

def load_kaggle(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['title', 'end_time', 'description', 'set_index', 'set_type', 'distance_miles', 'duration_seconds', 'muscle_group'])
    df = df.dropna(subset=['weight_lbs', 'reps'])
    df['weight_kg'] = df.pop('weight_lbs') * 0.453592 #keep measurements the same
    df.rename(columns={'start_time': 'date'}, inplace=True)

    #convert to date time and sort by date
    df['date'] = pd.to_datetime(df['date'], format="%d %b %Y, %H:%M")
    df = df.sort_values('date')
    
    return df



def load_hevy(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['title', 'end_time', 'description', 'superset_id', 'exercise_notes',
                              'set_index', 'set_type', 'distance_km', 'duration_seconds', 'rpe'])
    df = df.dropna(subset=['weight_kg', 'reps'])
    df.rename(columns={'start_time': 'date'}, inplace=True)

    #convert to date time and sort by date
    df['date'] = pd.to_datetime(df['date'], format="%b %d, %Y, %I:%M %p")
    df = df.sort_values('date')
    
    return df