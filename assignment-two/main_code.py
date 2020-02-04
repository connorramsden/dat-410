import numpy as np
import pandas as pd
from tensorflow import keras


def read_data(data_file):
    training_data = pd.read_csv(data_file)
    pred = 'your code here'
    return training_data, pred


def clean_data(train, pred):
    id_columns = ['encounter_id', 'icu_id', 'hospital_id', 'patient_id']
    numerical_data = train.select_dtypes(np.int64)
    numerical_data = numerical_data.drop(id_columns, axis=1)
    return numerical_data


training, prediction = read_data('data/training_v2.csv')

num_data = clean_data(training, prediction)

print(training.head())
