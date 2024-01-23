#!/usr/bin/env python

# Importing TensorFlow
import tensorflow as tf

# Loading the Fashion MNIST dataset from TensorFlow's datasets
data = tf.keras.datasets.fashion_mnist

# Splitting the dataset into training and testing sets
(training_images, training_labels), (test_images, test_labels) = data.load_data()
# The dataset is split into training images and labels, and test images and labels.

# Normalizing the training and test images
training_images = training_images / 255.0
test_images = test_images / 255.0
# Dividing by 255.0 to scale the pixel values to be between 0 and 1.

# Building the Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), # First layer that flattens the images to a 1D array
    tf.keras.layers.Dense(128, activation=tf.nn.relu), # Second layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # Output layer with 10 neurons (for each class) and softmax activation
])
# The model has three layers: a Flatten layer and two Dense layers.

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Compiling the model with 'adam' optimizer, 'sparse_categorical_crossentropy' loss function, and tracking 'accuracy' metric.

# Training the model
model.fit(training_images, training_labels, epochs=50)
# Training the model on the training images and labels for 50 epochs.


