from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np


iris = load_iris()
X = iris.data
y = iris.target

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DT
tree = DecisionTreeClassifier()
tree_param_grid = {"max_depth": [3, 5, 7],
                   "max_leaf_nodes": [None, 5, 10, 20, 30, 50]}

nb_cv = 3
tree_grid_search = GridSearchCV(tree, tree_param_grid)
tree_grid_search.fit(X_train, y_train)
print("DT Grid Search Best Parameters: ", tree_grid_search.best_params_)
print("DT Grid Search Best Accuracy: ", tree_grid_search.best_score_)

for mean_score, params in zip(tree_grid_search.cv_results_["mean_test_score"], tree_grid_search.cv_results_["params"]):
    print(f"Ortalama test score: {mean_score}, params: {params}")

cv_results = tree_grid_search.cv_results_
for i, params in enumerate(cv_results["params"]):
    print(f"Parameter {i}: {params}")
    for j in range(nb_cv):
        accuracy = cv_results[f"split{j}_test_score"][i]
        print(f"\tFold {j+1} - Accuracy: {accuracy}")