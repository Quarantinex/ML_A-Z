# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:07:31 2019

@author: Abhishek_Nayak1
"""
#part 1 (Compulsory - Introductory)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')

#print(dataset)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#print(X)
#print(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test)

#part 2 (Setting up the linear regression)
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#part 3 (Predicting the test set results)
Y_pred = regressor.predict(X_test)


#part 4 (Visualising the training set)
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()