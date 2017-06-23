#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:12:43 2017

@author: virginiacenisilva
"""

# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
adsSelected = []
numbersOfSelections = [0] * d
sumsOfRewards = [0] * d
totalReward = 0
for n in range(0, N):
    ad = 0
    maxUpperBound = 0
    for i in range(0, d):
        if (numbersOfSelections[i] > 0):
            averageReward = sumsOfRewards[i] / numbersOfSelections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbersOfSelections[i])
            upper_bound = averageReward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > maxUpperBound:
            maxUpperBound = upper_bound
            ad = i
    adsSelected.append(ad)
    numbersOfSelections[ad] = numbersOfSelections[ad] + 1
    reward = dataset.values[n, ad]
    sumsOfRewards[ad] = sumsOfRewards[ad] + reward
    totalReward = totalReward + reward

# Visualising the results
plt.hist(adsSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylable('Number of times each ad was selected')
plt.show()