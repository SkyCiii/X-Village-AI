
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

file_name = 'regression_exercise1_data.csv'
data = pd.read_csv(file_name)

data.insert(0, 'X_0', 1)

x_df = data.iloc[:, 0:2]
y_df = data.iloc[:, 2:3]

x_array = x_df.values
real_y_2D = y_df.values         # but this is fine.
real_y_1D = real_y_2D.ravel()   # 2D array -> 1D array

lr = LinearRegression()

lr.fit(x_array, real_y_1D)
predicted_y = lr.predict(x_array)

mse = mean_squared_error(real_y_1D, predicted_y)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_y_1D, predicted_y)
r2 = r2_score(real_y_1D, predicted_y)

print(rmse)

sgdr = SGDRegressor()

sgdr.fit(x_array, real_y_1D)
predicted_y2 = sgdr.predict(x_array)

mse2 = mean_squared_error(real_y_1D, predicted_y2)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(real_y_1D, predicted_y2)
r2_2 = r2_score(real_y_1D, predicted_y2)

print(rmse2)