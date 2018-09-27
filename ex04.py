from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import load_boston

boston = load_boston()

X = boston.data
y = boston.target

ridge = Ridge()
ridge.fit(X, y)
ridge_theta = ridge.coef_

lasso = Lasso()
lasso.fit(X, y)
lasso_theta = lasso.coef_

elasticnet = ElasticNet()
elasticnet.fit(X, y)
elasticnet_theta = elasticnet.coef_

print(ridge_theta, '\n', lasso_theta, '\n', elasticnet_theta)