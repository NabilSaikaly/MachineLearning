import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

df = pd.read_csv('china_gdp.csv')
print(df.head(5))

# Plotting the data set

plt.scatter(df.Year, df.Value, color = 'red')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()

# From the plot, we notice the relationship between the dependent and
# independent variable, can be modelled as Logisitc/Sigmoidal.
# So, We can choose the sigmoidal function as model
# for this non-linear regression model.


# Visualising a Logistic Function

x_data = np.arange(-5.0, 5.0, 0.1)
y_data = 1.0 / (1.0 + np.exp(-x_data))

plt.plot(x_data, y_data, '-r')
plt.xlabel(" X Data [INDEPENDENT VARIABLE] ")
plt.ylabel(" Y Data where y(x) is a Logistic Model. [DEPENDENT VARIABLE]")
plt.show()



# The formula of a Logistic Function contains several parameters.

# Modelling an Actual Logistic Function (Check Formula Picture)

def logistic(x_data, Beta_1, Beta_2):
	y_data = 1.0 / (1.0 + np.exp(-Beta_1*(x_data-Beta_2)))
	return y_data


# Let's give assumptions of a sigmoid/logistic line to see
# if it fits our actual data as a trendline.

Beta_1 = 0.1
Beta_2 = 1980.0

y_predidcted = logistic(df.Year, Beta_1, Beta_2)


# Visualising the trendline 


plt.scatter(df.Year, df.Value, color = 'red')
plt.plot(df.Year, y_predidcted*15000000000000, '-r')
plt.show()

# We multiplied the y_predicted by this number, to match the data.



# The task now is to find the best parameters Beta_1 and Beta_2
# to fit the model into the data.
# First, we need to normalize the data so that it appears similar 
# to all records and fields.


x_data = df.Year / max(df.Year)
y_data = df.Value / max(df.Value)

# Finding the best parameters for the fitline using curve_fit from scipy.optimize
# popt is the same as  parameters optimized


popt, pcov = curve_fit(logistic, x_data, y_data)

print("Sigmoid/Logistic Function Optimized Parameters: ")
print("Beta 1:",popt[0])
print("Beta 2:",popt[1])


# Now plotting the regression model against the data to confirm effectiveness.

x = np.linspace(1960, 2015, 55) #Creating x-axis data
x = x/max(x) #Normalizing 

y = logistic(x, *popt)
plt.plot(x_data, y_data, 'ro' , label = 'data')
plt.plot(x,y, linewidth=3.0, label = 'fit')
plt.legend(loc='best')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()


# Now let's create a non-linear model from a Train set
# and evaluate it using a test set, to determine the accuracy.

# Creating the Train/Test sets to ensure Out of Sample Policy.
msk = np.random.rand(len(df)) < 0.8
train_x = x_data[msk]
test_x = x_data[~msk]
train_y = y_data[msk]
test_y = y_data[~msk]

# Building the regression model.
popt, pcov = curve_fit(logistic, train_x, train_y)

# Predicting using the test set.
y_predicted = logistic(test_x, *popt)

# Evaluating the model accuracy.
print("MEAN ABSOLUTE ERROR: ", np.mean(np.absolute(y_predicted - test_y)))
print("MSE:" , np.mean((y_predicted - test_y) ** 2))
print("R2_Score:", r2_score(test_y,y_predicted) )



