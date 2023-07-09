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


# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(samplenetwork, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()
plt.close()

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


# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(samplenetwork, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()
plt.close()

# Sample input data
sample_X = np.array([[0], [1]])

# Predict output using the trained network
predicted_Y = predict(samplenetwork, sample_X)

print(predicted_Y)






