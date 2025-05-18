from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

# Read every 10th row from all columns
data_2nd = data.iloc[::10, :]

# Keep only the first two features and the target variable
X = data_2nd.iloc[:, :2] 
y = data_2nd.iloc[:, -1] 

reg = LinearRegression().fit(X, y)

# Get the coefficients and intercept
coefficients = reg.coef_
intercept = reg.intercept_

print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')

# Create a meshgrid for plotting
x1_range = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 100)
x2_range = np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
y_mesh = coefficients[0] * x1_mesh + coefficients[1] * x2_mesh + intercept

#3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x1_mesh, x2_mesh, y_mesh, color='b', alpha=0.5)

#3D Scatter
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, color='r')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
plt.show()
