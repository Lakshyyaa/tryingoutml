# Is there a relationship between the water temperature (deg C) and salinity? 
# Can we predict the Temperature of water (deg C) given the salinity?
# Predicting the Temperature using the salinity
# Temperature is the Response variable
# Salinity is the Predictor variable

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

# Dataset source: https://www.kaggle.com/datasets/sohier/calcofi
df=pd.read_csv('bottle.csv' ,low_memory=False)
pd.set_option('display.max_rows', None)
df=df[:510]
df.to_csv('new_file.csv', index=False)

# Creating a linear regression model with X as Salinity and Y as Temperature
# X = df[['Salnty']]
# Y = df[['T_degC']]
# model = LinearRegression()
# model.fit(X, Y)

# # Showing a scatter plot with the regression line
# plt.scatter(X, Y, alpha=0.5, label='Data')
# plt.plot(X, model.predict(X), color='red', label='Regression Line')
# plt.title('Scatter Plot with Linear Regression Line')
# plt.xlabel('Salinity')
# plt.ylabel('Temperature')
# plt.legend()
# plt.show()

# # Printing the parameters of the regression line
# slope = model.coef_[0]
# intercept = model.intercept_
# print("Slope:", slope[0])
# print("Intercept:", intercept[0])

# # Computing SSE, SST & SSR and printing them
# Y_pred = model.predict(X)
# # Calculate the residuals
# residuals = (Y - Y_pred).to_numpy()
# # Calculate SSE (Sum of Squares Residual)
# sse = np.sum(residuals**2)
# # Calculate SST (Total Sum of Squares)
# sst = np.sum((Y.to_numpy() - np.mean(Y.to_numpy()))**2)
# # Calculate SSR (Sum of Squares Explained)
# ssr = sst - sse
# print("SSE:", sse)
# print("SSR:", ssr)
# print("SST:", sst)