# Mnist_Data
A simple implementation of a deep neural network (DNN) to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. The model consists of a flattening layer, a hidden dense layer, and an output layer with 10 neurons for digit classification.
This is a Deep Neural Network (DNN), specifically a Multi-Layer Perceptron (MLP) model.

Key Characteristics:
Fully Connected Neural Network: Each neuron in a layer is connected to every neuron in the previous layer.
Flatten Layer: The input image (28x28 pixels) is flattened into a 1D vector before feeding it into the network.
Hidden Layer: The first hidden layer is a Dense layer with 128 neurons and uses the ReLU activation function to introduce non-linearity.
Output Layer: The final layer is a Dense layer with 10 neurons, each representing a class (digits 0-9). It uses the Softmax activation function to output the probabilities for each class.
Supervised Learning: The model is trained using labeled data (images of digits and their corresponding labels), which makes it a supervised learning approach.
Training & Evaluation: The model is trained for 5 epochs using the Adam optimizer and evaluated on the test set.
Purpose:
This model is trained to classify images from the MNIST dataset, which contains images of handwritten digits. The network predicts the class (digit) for a given input image, and the final output is the predicted label with the highest probability.
import tensorflow as tf  # Import TensorFlow library for deep learning
from tensorflow.keras.datasets import mnist  # Import MNIST dataset from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Load training and test data
x_train, x_test = x_train / 255, x_test / 255  # Normalize pixel values to be between 0 and 1

import matplotlib.pyplot as plt  # Import Matplotlib for plotting
plt.imshow(x_train[0])  # Display the first training image

from tensorflow import keras  # Import Keras API for building neural networks
from tensorflow.keras import layers  # Import Keras layers for building the model

# Define the neural network architecture
model = keras.Sequential([ 
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 image into a 1D array
    layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons and ReLU activation
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit) and softmax activation
])

# Compile the model with optimizer, loss function, and evaluation metrics
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model for 5 epochs using the training data
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)  # Print the accuracy on the test dataset

# Make predictions on the test data
predictions = model.predict(x_test)  
print(predictions[1])  # Print the predictions for the second test image
plt.imshow(x_test[1])  # Display the second test image

# Predict the label of the first test image and compare with the true label
predicted_label = model.predict(x_test[0].reshape(1, 28, 28))  # Reshape the first image and predict its label
print(f"Predicted label: {predicted_label.argmax()}")  # Print the predicted label (index of highest probability)
print(f"True label: {y_test[0]}")  # Print the true label of the first image
Here are some hashtags you can use:

#DeepLearning 
#NeuralNetwork 
#TensorFlow 
#Keras 
#MachineLearning 
#MNIST 
#DigitClassification 
#AI 
#ArtificialIntelligence 
#DataScience 
#Python 
#ComputerVision 
#MachineLearningModel
