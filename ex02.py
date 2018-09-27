import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt

# x = [[0.8, 0.2], [0.1, 0.3], [-0.5, -0.6]]
# y = [0.5, 0.3, -0.1]
# lr = LinearRegression()
# lr.fit(x, y)
# print(lr.coef_)
# print(lr.predict([[0.1, 0.2]]))

# LinearRegression

file_name = 'regression_exercise1_data.csv'
data = pd.read_csv(file_name)

data.insert(0, 'X_0', 1)

x_df = data.iloc[:, 0:2]
y_df = data.iloc[:, 2:3]

x_array = x_df.values
real_y_2D = y_df.values         # but this is fine.
real_y_1D = real_y_2D.ravel()   # 2D array -> 1D array

#

lr = LinearRegression()

lr.fit(x_array, real_y_1D)
predicted_y = lr.predict(x_array)

print(lr.coef_)

def plot_regression_line(x, real_y, predicted_y):
    
    plt.scatter(x['X_1'], real_y, s=60, alpha=.6)
    plt.plot(x['X_1'], predicted_y, 'r-')
    plt.show()

plot_regression_line(x_df, real_y_1D, predicted_y)

# SGDRegressor

sgdr = SGDRegressor()

sgdr.fit(x_array, real_y_1D)
predicted_y2 = sgdr.predict(x_array)

print(sgdr.coef_)

def plot_regression_line2(x, real_y, predicted_y2):
    plt.scatter(x['X_1'], real_y, s=60, alpha=.6)
    plt.plot(x['X_1'], predicted_y2, 'r-')
    plt.show()

plot_regression_line2(x_df, real_y_1D, predicted_y2)