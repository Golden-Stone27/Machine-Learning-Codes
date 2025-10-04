import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

# y = a0 + a1x --> Linear Regression
# y = a0 + a1x1 + a2x2 + ... + anxn --> Multi Variable Linear Regression
# y = a0 + a1x1 + a2x2 --> we will use this one

X = np.random.rand(100,2)
coef = np.array([3, 5])
# y = 0 + np.dot(X,coef)
y = np.random.rand(100) + np.dot(X,coef)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# ax.scatter(X[:,0], X[:,1], y)
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('y')

lin_reg = LinearRegression()
lin_reg.fit(X, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], y)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')

x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_pred = lin_reg.predict(np.array([x1.flatten(), x2.flatten()]).T)
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), alpha = 0.3)
plt.title("Multi Variable Linear Regression")

print("Katsayılar: ", lin_reg.coef_)
print("Kesişim: ", lin_reg.intercept_)
plt.show()
