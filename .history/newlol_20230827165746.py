import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
df=pd.read_csv('Weather.csv')
# predict max given minimum
# Assuming you want to predict MaxTemp based on MinTemp
X = df[['MinTemp']]
Y = df[['MaxTemp']]

# Create a LinearRegression model
model = LinearRegression()

# Fit the model
model.fit(X, Y)

# Predict using the model
predicted_max_temp = model.predict(X)

# Print first few predicted values
print(predicted_max_temp[:5])
