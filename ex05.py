from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV

boston = load_boston()


X = boston.data
y = boston.target

ridge = Ridge()

param_grid = {
    'alpha' : [1, 5, 10]
}

gs = GridSearchCV(ridge, param_grid)
gs.fit(X, y)

print(gs.best_estimator_)