from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# Create a data set

X = np.sort(5 * np.random.rand(80, 1), axis = 0) # Features
y = np.sin(X).ravel() # Targets
y[::2] += 0.5 * (0.5 - np.random.randn(40))



regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=15)
regr_1.fit(X, y)
regr_2.fit(X, y)

X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)


plt.figure()
plt.scatter(X, y, c='red', label='data')
plt.plot(X, y, c='red', label='data')
plt.plot(X_test, y_pred_1, color='blue', label='Max Depth: 2', linewidth=2)
plt.plot(X_test, y_pred_2, color='green', label='Max Depth: 5', linewidth=2)
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()


