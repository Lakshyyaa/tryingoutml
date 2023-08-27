import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
df=pd.read_csv('Weather.csv')
# predict max given minimum
X=df[['MaxTemp']]
Y=df[['MinTemp']]
model.fit(X, Y)

# Predict using the model
predicted_max_temp = model.predict(X)