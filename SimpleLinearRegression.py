import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score


df = pd.read_csv('FuelConsumptionCo2.csv')
print(df.head())
print(df.describe())

# Extracting some features to explore 
# 'ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS'

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(5))

cdf.hist()
plt.show()

# Now, let's plot each of these features against the Emission,
# to see how linear their relationship is:
# Emissions vs. Engine Size

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('Emissions')
plt.show()


# Emissions vs. Cylinders

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color ='blue')
plt.xlabel('Cylinders')
plt.ylabel('Emissions')
plt.show()


# Emissions vs. FUELCONSUMPTION_COMB

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color ='blue')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('Emissions')
plt.show()


# Creating Train/Test Split Test.
# Train set is 80% of the dataframe
# Test set is 20% of the data frame
# We will create a mask that will randomly pick 80% and 20% of the DF.

msk = np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk]


# Creating a Simple Regression model from training the train set!
# The independent variable will be the Engine Size
# Visualizing the train set distribution

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color ='blue')
plt.xlabel('Engine Size TRAIN SET')
plt.ylabel('Emissions TRAIN SET')
plt.show()


# Modelling the data based on the train set that we chose!

reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']]) #Specifying the indep. variable
train_y = np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(train_x, train_y)
print("Regression Coefficient: ", reg.coef_)
print("Regression Intercept: ", reg.intercept_)


# Visualizing the regression line on the training data set
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, reg.coef_[0][0]*train_x + reg.intercept_[0], '-r')
plt.xlabel('Engine Size TRAIN SET')
plt.ylabel('Emissions TRAIN SET')
plt.show()


# # MKN SURE  Visualizing the regression line on the training data set
# plt.scatter(train_x, train.CO2EMISSIONS, color='blue')
# plt.plot(train_x, reg.coef_[0][0]*train_x + reg.intercept_[0], '-r')
# plt.xlabel('Engine Size TRAIN SET train_X')
# plt.ylabel('Emissions TRAIN SET')
# plt.show()


'''
  Evaluating the model by checking the accuracy or the error that exists
  Between the predicted value and actual value.
  We will do the testing now. the test set is masked as 20% of the whole DF
  But we need to specify which indepdent variable! 
  Then we calculate a specific metric to evaluate the model!
  Metrics to calculate: Mean Absolute Error, MSE, r2-score
'''

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']]) #This is the actual value
test_y_ = reg.predict(test_x) #This is the predicted value from the REG FIT.
'''test_y_ from reg.predict(test_x) is the same as:
   test_y_ = test_x*reg.coef_[0][0] + reg.intercept_[0]'''
print("\nEvaluating the Regression model: ")
print("Mean Absolute Error: ", np.mean(np.absolute(test_y - test_y_)))
print("Residual Sum of squares (MSE): ", np.mean((test_y - test_y_)**2))
print("R2 Score: ", r2_score(test_y, test_y_))



# Check the difference of model performance when the independent variable
# is the FUELCONSUMPTION_COMB instead of ENGINESIZE
# finding train_x, [train_y remains the same!]
# finding test_x, [test_y remains the same!]
# only test_y_ changes! as it is the new prediction.

train_x = train[['FUELCONSUMPTION_COMB']]
test_x = test[['FUELCONSUMPTION_COMB']]
reg2 = linear_model.LinearRegression()
reg2.fit(train_x, train_y)
print("\n\nRegression 2 Coefficient: ", reg2.coef_)
print("Regression 2 Intercept: ", reg2.intercept_)

print("\nEvaluating the model of Regression 2:")
test_y_ = reg2.predict(test_x)
print("Mean Absolute Error: ", np.mean(np.absolute(test_y - test_y_)))
print("Residual Sum of squares (MSE): ", np.mean((test_y - test_y_)**2))
print("R2 Score: ", r2_score(test_y, test_y_))
