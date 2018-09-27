
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name = 'regression_exercise1_data.csv'
data = pd.read_csv(file_name)

data.insert(0, 'X_0', 1)

x = data.iloc[:, 0:2]
y = data.iloc[:, 2:3]

x_array = x.values
y_array = y.values

theta_1 = np.transpose(np.array([[0, 0]]))
theta_2 = np.transpose(np.array([[1, 1]]))
theta_3 = np.transpose(np.array([[10, -1]]))

def h(x, theta):
    return np.dot(x_array, theta)

def compute_cost(x, y, theta):
    error = np.power((h(x, theta)-y), 2)
    error_sum = np.sum(error)
    lenth = 2*len(y_array)
    J = error_sum/lenth
    print(J)

def plot_input_data(x, y_array):
    plt.scatter(x['X_1'], y_array, s=60, alpha=.6)
    plt.xlabel("X_1")
    plt.ylabel("y")
    plt.show()

def plot_regression_line(x, y_array, theta):
    plt.scatter(x[:,1], y, s=60, alpha=.6)
    plt.plot(x[:,1], theta[0] + theta[1]*x[:,1], 'r-')
    plt.xlabel("X_1")
    plt.ylabel("y")
    plt.show()

# compute_cost(x_array, y_array, theta_1)
plot_regression_line(x_array, y_array, theta_1)
plot_regression_line(x_array, y_array, theta_2)
plot_regression_line(x_array, y_array, theta_3)