import numpy as np
import random
import cv2 as cv


# class baseLayer():
#     def __init__(self):
#         self.input=None
#         self.output=None
    
#     def forward(self , input):
#         pass

#     def backward(self ,output_gradiebnt , learning_rate ):
#         pass

input = np.random.randn(10)
input=input.T

class denseLayer():

    #Initializes weights and biases values randomly
    def __init__(self , input_size , output_size):
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size)
    
    def forward(self , input):
        self.input = input
        return np.dot(self.weights , self.input) + self.bias

    def backward(self , output_gradient , learning_rate):
        pass

    def activationFunction(z):
        # return np.maximum(z, 0)
        return z>0

#hidden layer 1
hidden_1 = denseLayer(10,10)
z1=hidden_1.forward(input)

#hidden layer 2
hidden_2 = denseLayer(10,10)
z2 = hidden_2.forward(z1)


layer_1_activated= denseLayer.activationFunction(z1)
layer_2_activated= denseLayer.activationFunction(z2)

print("layer 1 output is:\n",z1)
print("layer 1 Activation:\n",layer_1_activated)

print("\nlayer 2 output is:\n",z2)
print("layer 2 Activation:\n",layer_1_activated)

