import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
df=pd.read_csv('Weather.csv')

# Is there a relationship between the daily minimum and maximum temperature? Can you predict the maximum temperature given the minimum temperature?
# Predicting the max temp using the minimum temp
# Max Temperature is the Response variable
# Min Temperature is the Predictor variable

X = df[['MinTemp']]
Y = df[['MaxTemp']]
model = LinearRegression()
model.fit(X, Y)

# Showing a scatter plot with the regression line
plt.scatter(X, Y, alpha=0.5, label='Data')
plt.plot(X, model.predict(X), color='red', label='Linear Regression')
plt.title('Scatter Plot with Linear Regression Line')
plt.xlabel('Min Temperature')
plt.ylabel('Max Temperature')
plt.legend()
plt.show()

# Printing the parameters of the regression line
slope = model.coef_[0]
intercept = model.intercept_
print("Slope:", slope)
print("Intercept:", intercept)



# 
Y_pred = model.predict(X)

# Calculate the residuals
residuals = Y - Y_pred

# Calculate SSE (Sum of Squares Residual)
sse = np.sum(residuals**2)

# Calculate SST (Total Sum of Squares)
sst = np.sum((Y - np.mean(Y))**2)

# Calculate SSR (Sum of Squares Explained)
ssr = sst - sse

# print("SSE:", sse)
# print("SSR:", ssr)
# print("SST:", sst)
# //notes
# ONE OF THE BELOW IS INDEXED OTHER IS NOT
# print(Y)
# print(Y_pred)
print(residuals)