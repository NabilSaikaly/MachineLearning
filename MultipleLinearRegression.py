import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv('FuelConsumptionCo2.csv')
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# Creating a Multiple Linear Regression Model.
# multiple linear regression here is predicting CO2EMISSION using several features.
# the features are FUELCONSUMPTION_COMB, ENGINESIZE and CYLINDERS of cars. 

# Splitting the Train/Test sets: 80% for Train set and 20% for test set.
msk = np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk]

# Creating the Regression Model
reg3 = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
reg3.fit(train_x, train_y)

print("The Multiple Linear Regression Coefficients are: ", reg3.coef_)

# Testing the Regression Model with out-of-sample data set.

test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
y_hat = reg3.predict(test_x)
print("Residual sum of squares: ", np.mean((y_hat - test_y) ** 2))
print('Variance score: ', reg3.score(test_x, test_y))

# Creating a Multiple Linear Regression Model, with more features now (more Independent Variables).
# the features are FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY, EngineSize and Cylinders of cars. 

# Creating the Regression Model (Training)
reg4 = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
reg4.fit(train_x, train_y)
print("\nThe Multiple Regression Coefficients are: ", reg4.coef_)

# Evaluating the Regression Model (Testing)
test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
y_hat = reg4.predict(test_x)
print("Residual sum of squares: ", np.mean((y_hat - test_y) ** 2))
print('Variance score: ', reg4.score(test_x, test_y))





