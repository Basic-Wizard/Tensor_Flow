#!/usr/bin/env python

# Importing necessary libraries
# TensorFlow, for building and training neural network models
import tensorflow as tf         
# NumPy, for numerical operations
import numpy as np              

# Importing specific modules from TensorFlow
# Sequential model from Keras API in TensorFlow
from tensorflow.keras import Sequential
# Dense layer, a standard layer type in neural networks  
from tensorflow.keras.layers import Dense 


# Building the Neural Network model
# Creating a Sequential model with one Dense layer
model = Sequential([Dense(units = 1, input_shape = [1])]) 
# 'units = 1' means the layer has one neuron.
# 'input_shape = [1]' signifies that the input to this layer is one-dimensional.


# Compiling the model
model.compile(optimizer='sgd', loss='mean_squared_error')
# 'optimizer='sgd'' means that Stochastic Gradient Descent will be used as the optimizer for the function.
# this will take the loss from the previous guesses and use them to generate the next set of guesses 
# 'loss ='mean_squared_error'' means that the mean squared error will be used to calculate the loss for each epoch.

# Preparing the training data
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float) # Input data
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float) # Output data
# The relationship in this data is y = 2x - 1.

# Training the model
# Training the model using the input-output pairs.
# the fit method takes the imput training data as the first argument and the output data as the second argument
model.fit(x, y, epochs=500)
# 'epochs = 500' means the model iterates over the data 500 times (epochs) to adjust its internal parameters.

# Making a prediction
print(model.predict([10.0]))
# ^ Predicting output using the trained model for a new input value [10.0].