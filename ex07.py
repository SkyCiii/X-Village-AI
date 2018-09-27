import csv
import pandas as pd








filename = 'train_data.csv'
data = pd.read_csv(filename)
print(data.values)

