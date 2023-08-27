import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
df=pd.read_csv('Weather.csv')
# predict max given minimum
X = df[['MinTemp']]
Y = df[['MaxTemp']]
# model = LinearRegression()
# model.fit(X, Y)
# predicted_max_temp = model.predict(X)
# print(predicted_max_temp[:5])
df.set_index
