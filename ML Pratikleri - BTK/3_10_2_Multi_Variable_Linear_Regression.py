from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(train_x, train_y)
y_pred = lin_reg.predict(test_x)

rmse = np.sqrt(mean_squared_error(test_y, y_pred))
print("rmse: ", rmse)
