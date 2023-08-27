import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
df=pd.read_csv('Weather.csv')

# Predicting the max temp using the minimum temp
# Max Temperature is the Response variable
# Min Temperature is the Predictor variable

X = df[['MinTemp']]
Y = df[['MaxTemp']]
model = LinearRegression()
model.fit(X, Y)
# Plot the data points
plt.scatter(X, Y, alpha=0.5, label='Data')
# Plot the linear regression line
plt.plot(X, model.predict(X), color='red', label='Linear Regression')
plt.title('Scatter Plot with Linear Regression Line')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.legend()
plt.show()