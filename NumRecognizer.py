import numpy as np
import random

class NeuralNetwork:
    def inputLayer (self, input_size):
        self.inputValues = np.random.randn(input_size)
        return self.inputValues
    
    def createLayer (self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
       
    def sigmoid(self , x):
        return 1/(1+np.exp(-x))
    
    def lossFunction(self, y_pred , y_exp):
        return np.mean((y_pred - y_exp)**2)
    
#Created instance of the class
NN = NeuralNetwork()

#Input Layer
inputs = [-1.53132339, 3.19740662]
print("Input Layer Values:\n", inputs)

#Hidden Layer
NN.createLayer(2,3)
z1 = NN.forward(NN.inputLayer(2))
a1 = NN.sigmoid(z1)
print("\nHidden Layer Sigmoid:\n", a1)

#Output Layer
NN.createLayer(3,2)
z2 = NN.forward(a1)
a2 = NN.sigmoid(z2)
print("\nOutput Layer Sigmoid:\n", a2)

#Loss/Cost Calculation
expectedOutput = [1,0]
cost = NN.lossFunction(a2, expectedOutput)
print("\nTotal cost is:", cost)

#Calculate Gradients
