"""This is the Hello World of Neural Networks.
It finds the pattern associated between the x and y data sets
"""

__author__ = 'Connor Ramsden'
__version__ = '0.1'
__date__ = 'January 29th, 2020'

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='adam', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=5000)

print(model.predict([10.0]))
