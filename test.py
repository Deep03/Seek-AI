import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(self.num_layers):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward(self, X):
        self.activations = []
        self.inputs = [X]

        # Perform forward propagation
        for i in range(self.num_layers):
            activation = self.sigmoid(np.dot(self.inputs[i], self.weights[i]) + self.biases[i])
            self.activations.append(activation)
            self.inputs.append(activation)

        return self.activations[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Compute gradients for the output layer
        delta = self.activations[-1] - y
        dW = np.dot(self.activations[-2].T, delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m
        gradients = [(dW, db)]

        # Compute gradients for the hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * self.sigmoid_derivative(self.inputs[i + 1])
            dW = np.dot(self.inputs[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients.insert(0, (dW, db))

        # Update weights and biases using gradients
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)

            # Backward propagation
            self.backward(X, y, learning_rate)

            # Compute and print the loss
            loss = self.cross_entropy_loss(output, y)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def predict(self, X):
        # Predict the output for new inputs
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_probs) / m
        return loss


# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Preprocess the input data
X /= 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoded vectors
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]

# Create an instance of the neural network
input_size = X_train.shape[1]
hidden_sizes = [128, 64]  # Customize the hidden layer sizes as needed
output_size = num_classes
nn = NeuralNetwork(input_size, hidden_sizes, output_size)

# Train the neural network
epochs = 1000  # Adjust the number of epochs as needed
learning_rate = 0.1  # Adjust the learning rate as needed
nn.train(X_train, y_train_one_hot, epochs, learning_rate)

# Use the trained model to make predictions on the test set
y_pred = nn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

