# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:38:19 2019

@author: Abhishek_Nayak1
"""

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Random Selection
#Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    
#Visualising the result in Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times such ad was selected')
plt.show()

#Upper Confidence Bound
#Implementing Upper Cofidence Bound
ads_select = []
numbers_of_selections =  [0] * d
sums_of_rewards = [0] * d
total_rewards = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_select.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    rewards = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + rewards
    total_rewards = total_rewards + rewards

#Visualising the result in Histogram
plt.hist(ads_select)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times such ad was selected')
plt.show()