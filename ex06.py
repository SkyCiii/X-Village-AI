from sklearn import datasets
from sklearn.model_selection import train_test_split

wine = datasets.load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state = 0)