from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

# Read every 10th row from all columns
data_10th = data.iloc[::10, :]

X = data_10th.iloc[:, :-1] 
y = data_10th.iloc[:, -1]  

reg = LinearRegression().fit(X, y)

# Get the coefficients and intercept
coefficients = reg.coef_
intercept = reg.intercept_

print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')

# Predict the Median House Value
input_data = [[8.3153, 41.0, 6.894423, 1.053714, 323.0, 2.533576, 37.88, -122.23]]
predicted_value = reg.predict(input_data)

print(f'Predicted Median House Value: {predicted_value[0]}')

