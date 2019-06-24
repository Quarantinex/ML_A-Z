# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:20:38 2019

@author: Abhishek_Nayak1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
print(dataset)

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
print(transactions)