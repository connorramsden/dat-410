import numpy as np
import pandas as pd
from tensorflow import keras

training = pd.read_csv('data/training_v2.csv', nrows=100)


def hospital_model(y_new):
    model = keras.Sequential(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model.predict(y_new)


prediction = hospital_model([7.0])
print(prediction)
