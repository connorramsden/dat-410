""" A linear regression model between the number of bedrooms a house has and its price
"""

__author__ = 'Connor Ramsden'
__version__ = '0.1'
__date__ = 'January 30th, 2020'

import numpy as np
from tensorflow import keras


# GRADED FUNCTION: house_model
def house_model(y_new):
    # should return the predicted value of y
    # The number of rooms in the given house
    num_rooms = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
    # The price of the house (in hundreds of thousands)
    house_price = np.array([1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], dtype=float)
    # Build a model, train it, return the predicted value
    model = keras.Sequential(keras.layers.Dense(1))  # build our model
    model.compile(optimizer='adam', loss='mean_squared_error')  # compile the model w/ Adam optimizer and MSE loss algo.
    model.fit(num_rooms, house_price, epochs=10000)  # send the model running. I'm GPU training so 2500 epoch is fast
    return model.predict(y_new)  # return the model's predicted data


prediction = house_model([7.0])  # Predicting for a 7 bedroom house
print(prediction)  # Printing prediction to console
