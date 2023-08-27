# import pandas as pd 
# import numpy as np 
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression 
# df=pd.read_csv('Weather.csv')

# # Predicting the max temp using the minimum temp
# # Max Temperature is the Response variable
# # Min Temperature is the Predictor variable

# X = df[['MinTemp']]
# Y = df[['MaxTemp']]
# model = LinearRegression()
# model.fit(X, Y)

# # Showing a scatter plot with the regression line
# plt.scatter(X, Y, alpha=0.5, label='Data')
# plt.plot(X, model.predict(X), color='red', label='Linear Regression')
# plt.title('Scatter Plot with Linear Regression Line')
# plt.xlabel('Min Temperature')
# plt.ylabel('Max Temperature')
# plt.legend()
# plt.show()

# # Printing the parameters of the regression line
# slope = model.coef_[0]
# intercept = model.intercept_
# print("Slope:", slope)
# print("Intercept:", intercept)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('Weather.csv')

# Define your custom cost function
def custom_cost_function(y_actual, y_predicted):
    return np.mean((y_actual + y_predicted)**2)

# Define gradient descent optimizer
def gradient_descent(X, y_actual, learning_rate, num_iterations):
    n = len(y_actual)
    theta = np.zeros((2, 1))  # Initialize parameters (slope and intercept)
    
    for _ in range(num_iterations):
        y_predicted = np.dot(X, theta)
        gradient = np.dot(X.T, (y_predicted + y_actual)) / n
        theta -= learning_rate * gradient
    
    return theta

# Prepare the data
X = df[['MinTemp']].values
Y = df[['MaxTemp']].values

# Add a column of ones to X for intercept term
X = np.c_[np.ones((X.shape[0], 1)), X]

# Use gradient descent to optimize parameters
learning_rate = 0.001
num_iterations = 1000
optimized_params = gradient_descent(X, Y, learning_rate, num_iterations)

# Extract slope and intercept from optimized parameters
intercept = optimized_params[0][0]
slope = optimized_params[1][0]

# Plot the scatter plot with the regression line
plt.scatter(X[:, 1], Y, alpha=0.5, label='Data')
plt.plot(X[:, 1], X.dot(optimized_params), color='red', label='Custom Regression')
plt.title('Scatter Plot with Custom Regression Line')
plt.xlabel('Min Temperature')
plt.ylabel('Max Temperature')
plt.legend()
plt.show()

# Print the parameters of the custom regression line
print("Slope:", slope)
print("Intercept:", intercept)
