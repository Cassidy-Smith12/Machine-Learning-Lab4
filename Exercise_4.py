import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

data_dropped = data.drop(columns=['Longitude', 'Latitude'])

#Pairplot
sns.pairplot(data_dropped)
plt.show()
