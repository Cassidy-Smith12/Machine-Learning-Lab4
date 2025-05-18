from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]  

# Scale the data using Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reg = LinearRegression().fit(X_scaled, y)

# Get the coefficients
coefficients = reg.coef_

# Find the coefficient with the most weight (absolute value)
max_weight_index = abs(coefficients).argmax()
max_weight_feature = X.columns[max_weight_index]

print(f'The feature with the most weight is: {max_weight_feature}')
