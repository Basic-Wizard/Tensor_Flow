#!/usr/bin/env python

# Importing TensorFlow
import tensorflow as tf

# Loading the Fashion MNIST dataset from TensorFlow's datasets
data = tf.keras.datasets.fashion_mnist

# Splitting the dataset into training and testing sets
(training_images, training_labels), (test_images, test_labels) = data.load_data()
# The dataset is split into training images and labels, and test images and labels.

# Reshaping and normalizing the training images
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
# Reshaping training images to [60000, 28, 28, 1] for Conv2D layer and normalizing by dividing by 255.

# Reshaping and normalizing the test images
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0
# Reshaping test images to [10000, 28, 28, 1] for Conv2D layer and normalizing by dividing by 255.

# Building the Convolutional Neural Network (CNN) model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
# The model includes Conv2D and MaxPooling layers for feature extraction, followed by Dense layers for classification.

# Compiling the CNN model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Compiling the model with the 'adam' optimizer and 'sparse_categorical_crossentropy' as the loss function.

# Training the CNN model
model.fit(training_images, training_labels, epochs=50)
# Training the model on the training images and labels for 50 epochs.

# Evaluating the CNN model
model.evaluate(test_images, test_labels)
# Evaluating the model's performance using the test images and test labels.

# This script sets up and trains a convolutional neural network using the Fashion MNIST dataset, 
# which is useful for image classification tasks.
