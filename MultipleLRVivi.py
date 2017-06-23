#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:28:16 2017

@author: virginiacenisilva
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independant Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoderx = LabelEncoder()
x[:,3] = labelencoderx.fit_transform(x[:,3])
# Dummy Encoding
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding the Dummy Variable Trap
x = x[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
xTrain = scx.fit_transform(xTrain)￼

xTest = scx.transform(xTest)
'''
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

# Predicting the Test set results
yPred = regressor.predict(xTest)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
xOpt = x[:, [0,1,2,3,4,5]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit()

# Look for the predictor with the highest P value
regressorOLS.summary()

# Remove the index 2 with highest p value
xOpt = x[:, [0,1,3,4,5]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit()
regressorOLS.summary()

xOpt = x[:, [0,3,4,5]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit()
regressorOLS.summary()

xOpt = x[:, [0,3,5]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit()
regressorOLS.summary()

xOpt = x[:, [0,3]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit()
regressorOLS.summary()