import numpy as np
import pandas as pd
from tensorflow import keras
import sklearn.preprocessing as prep


# Read in CSV files
def read_data(training_file, predictions_file):
    training_data = pd.read_csv(training_file)
    prediction_data = pd.read_csv(predictions_file)
    return training_data, prediction_data


# Clean up training and prediction data
def clean_data(training, predictions):

    # Identify training features & eliminate unwanted data
    categorical_data = list(training.select_dtypes('object').columns)
    generic_data = ['encounter_id', 'patient_id', 'hospital_id']
    target_data = ['hospital_death']
    exclude_cols = categorical_data + generic_data + target_data
    numerical_data = [col for col in training.columns if col not in exclude_cols]

    # Set up training and prediction targets
    training_target = training[target_data]
    training_features = training[numerical_data]
    prediction_features = predictions[numerical_data]

    # Replace missing values with 0's
    training_features.fillna(0, inplace=True)
    prediction_features.fillna(0, inplace=True)

    # scale data, turn DataFrames to numpy arrays
    data_scaler = prep.StandardScaler(copy=True, with_mean=True, with_std=True)
    data_scaler.fit(training_features)
    training_features = data_scaler.transform(training_features)
    prediction_features = data_scaler.transform(prediction_features)

    return training_target, training_features, prediction_features


# Build model from training data
def build_model(training_features, training_target):
    model = keras.Sequential([keras.layers.Dense(units=174, activation='tanh', input_shape=[174]),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(1, activation='tanh')])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(training_features, training_target.values, epochs=5)
    return model


def main():
    training_data = 'data/training_v2.csv'
    prediction_data = 'data/unlabeled.csv'

    train, pred = read_data(training_data, prediction_data)
    training_targ, training_feat, prediction_feat = clean_data(train, pred)

    trained_model = build_model(training_feat, training_targ)
    out_prediction = trained_model.predict(prediction_feat)
    print(out_prediction)


main()
