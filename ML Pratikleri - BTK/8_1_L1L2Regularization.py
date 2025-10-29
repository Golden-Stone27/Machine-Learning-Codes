from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge
ridge = Ridge()
ridge_param_grid = {"alpha": [0.1, 1, 10, 100]}

ridge_grid_search = GridSearchCV(ridge, param_grid=ridge_param_grid, cv = 5)
ridge_grid_search.fit(train_x, train_y)
print("Ridge en iyi parameters: ", ridge_grid_search.best_params_)
print("Ridge en iyi score: ", ridge_grid_search.best_score_)

best_ridge_model = ridge_grid_search.best_estimator_
y_pred_ridge = best_ridge_model.predict(test_x)
ridge_mse = mean_squared_error(test_y, y_pred_ridge)
print("Ridge mse: ", ridge_mse)

print()
# Lasso
lasso = Lasso()
lasso_param_grid = {"alpha": [0.1, 1, 10, 100],}

lasso_grid_search = GridSearchCV(lasso, param_grid=lasso_param_grid, cv = 5)
lasso_grid_search.fit(train_x, train_y)
print("Lasso en iyi parameters: ", lasso_grid_search.best_params_)
print("Lasso en iyi score: ", lasso_grid_search.best_score_)

best_lasso_model = lasso_grid_search.best_estimator_
y_pred_lasso = best_lasso_model.predict(test_x)
lasso_mse = mean_squared_error(test_y, y_pred_lasso)
print("Lasso mse: ", lasso_mse)