import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

data = np.array([[i, 2 * i, np.random.uniform(-100, 100)] for i in range(100)])
fig = plot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plot.show()

center = data - data.mean()
covmat = np.cov(data, rowvar=False)

vals, vects = np.linalg.eig(covmat)

v1 = vects[1].reshape(3, -1)
v2 = vects[2].reshape(3, -1)

c1 = np.dot(center, v1).reshape(1, -1)
c2 = np.dot(center, v2).reshape(1, -1)

plot.figure()
plot.scatter(c1, c2, c='black')
plot.show()