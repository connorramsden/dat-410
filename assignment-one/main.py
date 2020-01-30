import tensorflow as tf
import numpy as np
from tensorflow import keras


# GRADED FUNCTION: house_model
def house_model(y_new):
    # should return the predicted value of y
    # Make some data
    # Build a model, train it, return the predicted value
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='adam', loss='mean_squared_error')
    num_rooms = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
    house_price = np.array([100, 150, 200, 250, 300, 350, 400], dtype=int)
    model.fit(num_rooms, house_price, epochs=5000)
    return model.predict(y_new)


prediction = house_model([7.0])
print(prediction)
