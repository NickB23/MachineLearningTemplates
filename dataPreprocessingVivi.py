# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
# : refers to the columns of th data - he wants the three colums of the independent variables
X = dataset.iloc[:,:-1].values
# for y we only want the colums of the dependent variable
y = dataset.iloc[:,3].values

# Missing data
# Replace the missing data by the mean of the values from the column where the data missing
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X [:, 1:3])

# Categorical data
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoderX = LabelEncoder()
X[:,0] = labelencoderX.fit_transform(X[:,0])
# We have to prevent that the machine thinks that the categories are greater than another based on the number they are assigned
# To prevent this we use dummy variables

# Dummy Encoding
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Because the purchased column in the dependent variable we don't need to transform it into dummy variable
labelencoderY = LabelEncoder()
y = labelencoderY.fit_transform(y)


#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
# sc = scale
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

                       



