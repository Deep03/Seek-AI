import numpy as np    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from basicNN import *

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

samplenetwork = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# train
train(samplenetwork, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)



# Sample input data
sample_X = np.array([[1], [0]])

# Predict output using the trained network
predicted_Y = predict(samplenetwork, sample_X)

print(predicted_Y)






