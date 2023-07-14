import numpy as np
import random

class Layer:
    def initParams(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size)
        return self.weights, self.bias
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def inputInitializer(self, input_size):
        self.input = np.random.randn(input_size)
        return self.input
    
    def ActivationReLU(self, z):
        return np.maximum(0, z)
    
    def ActivationSoftmax(self, Z):
        A = np.exp(Z) / np.sum(np.exp(Z))
        return A


layer = Layer()

input_neurons = layer.inputInitializer(4)
w1, b1 = layer.initParams(4, 8)
z1 = layer.forward(input_neurons)
z1_relu = layer.ActivationReLU(z1)

w2, b2 = layer.initParams(8, 4)
y = layer.forward(z1_relu)
y_softmax = layer.ActivationSoftmax(y)

print('-' * 10, "Neural Network Tester", '-' * 30)
print("\nInput Layer:\n", input_neurons)
print("\nHidden Layer One:\n", "\nweights:\n", w1, "\n\nbiases:\n", b1)
print("\nReLU Output:\n", z1_relu)
print("\nOutput Layer:\n", "\nweights:\n", w2, "\n\nbiases:\n", b2)
print("\nSoftmax Output:\n", y_softmax)
