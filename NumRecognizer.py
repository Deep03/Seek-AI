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
    
    def softmax(self, Z):
        A = np.exp(Z) / np.sum(np.exp(Z))
        return A
    
    def sigmoid(self , x):
        return 1/(1+np.exp(-x))
    

def NodeCost(z2_softmax , expOutput):
    error = expOutput - z2_softmax
    return error**2
def NodeCostDerivative(z2_softmax , expOutput):
    return 2*(z2_softmax - expOutput)

def sigmoidDerivattive(x):
    dev = layer.sigmoid(x)
    return x * (1 - x)

def mse(x, y):
    return np.mean(np.power(x- y, 2))
    
def derv(self):
    pass  

#Created instance of the class
layer = Layer()
inputs =  [ 0.68544165 ,
          -0.40785542]

w1, b1 = layer.initParams(2, 3)
z1 = layer.forward(inputs)
z1_A = layer.sigmoid(z1)

w2, b2 = layer.initParams(3, 2)
z2 = layer.forward(z1_A)
z2_softmax = layer.softmax(z2)

expOutput = [1,0]

cost = mse(z2_softmax,expOutput)


# print("Inputs:\n", inputs)
# print("\nHidden Layer:-\n", "weights:\n", w1, "\nbiases:\n", b1)
# print("\nSigmoid Output:\n",z1)
# print("\nOutput Layer:\n", "weights:\n", w2, "\nbiases:\n", b2)
print("\nSoftmax Output:\n",z2_softmax)
print("\nExpected Output:", expOutput)
print("\nCost is:", cost)
