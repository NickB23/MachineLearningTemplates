#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:21:15 2017

@author: virginiacenisilva
"""

# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import random
N = 10000
d = 10
adsSelected = []
numbersOfRewards_1 = [0] * d
numbersOfRewards_0 = [0] * d
totalReward = 0
for n in range(0, N):
    ad = 0
    maxRandom = 0
    for i in range(0, d):
        randomBeta = random.betavariate(numbersOfRewards_1[i] + 1, numbersOfRewards_0[i] + 1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            ad = i
    adsSelected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1: 
        numbersOfRewards_1[ad] = numbersOfRewards_1[ad] + 1
    else: 
        numbersOfRewards_0[ad] = numbersOfRewards_0[ad] + 1
    totalReward = totalReward + reward

# Visualising the results
plt.hist(adsSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylable('Number of times each ad was selected')
plt.show()