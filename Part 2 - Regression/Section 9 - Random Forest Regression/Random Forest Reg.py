# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:08:35 2019

@author: Abhishek_Nayak1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
print(dataset)

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values
print(X)
print(Y)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, Y)

Y_pred = regressor.predict([[6.5]])
print(Y_pred)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random forest Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()