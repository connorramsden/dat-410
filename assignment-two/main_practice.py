import pandas as pd


def data_reading(data_set):
    train = pd.read_csv(data_set)
    pred = 'your code here'  # put your code for reading predictions data in the unlabeled .csv file
    return train, pred


training, predictions = data_reading('data/training_v2.csv')

# define id columns
id_columns = ['encounter_id', 'icu_id', 'hospital_id']

training.drop(id_columns, axis=1)

# Select numerical data from entire set
numeric_col = training.select_dtypes('Int64')

print('')
