# Machine Learning 
I created this repository to share all algorithms/techniques I am learning in the Machine Learning Field from various sources (Technical Documentations, Online Courses, Professional Trainings and more).
This repository will be continiously enriched because I am still learning ML Algorithms and techniques.

## ML Techniques Coverage:

My Learning path in both supervised and unsupervised machine learning:
1. Regression/Estimation (Covered)
2. Classification
3. Associations
4. Clustering
5. Recommendation Systems
6. Sequence Mining
7. Anomaly Detection
8. Dimension Reduction

## Regression Examples:
Source: IBM Machine Learning Course with Python on Coursera.
ML Regression will be used to predict/estimate the fuel consumption and CO2 Emissions of cars. A data relating CO2 Emissions and Fuel Consumption (FuelConsumptionCo2.csv) of already produced cars will be evaluated. The data will be splitted into training set and test set. We will train our ML Model using the training set, then evaluate the accuracy of the trained model using the test set. Once the required accuracy is obtained, the model will be used to predict/estimate an unknown value. 
The ML will be done using three different regression techniques: Simple Linear, Multiple Linear and Polynomial Regression.
Then a Non-Linear Regression model will be fitted to China's GDP from 1960 to 2014 data.

### Regression Example: Simple Linear Regression
Explanation is available as comments in SLReg.py
### Regression Example: Multiple Linear Regression
Explanation is available as comments in MLReg.py
### Regression Example: Polynomial Regression
Further explanation than the one available as comments in PolynomialRegression.py:
Whenever the data between the independent variable and dependent variable is shaped in a way
that is not linear, YET if the function/relationship  can be modelled as polynomial function,
(i.e.) we can represent it in the form of (a_n)(x^n) + (a_n-1)(x^n-1) + ... + (a_1)(x^1) + (a_0)
we can use the Polynomial function and transform/express it to/as a multiple linear regression
And by that we would use the multiple linear regression training/testing procedures in order
to solve a polynomial regression problem.
Transformation is in the following:
Suppose the Polynomial is P(x) = (a3)(x^3) + (a2)(x^2) + (a1)(x) + (a4)
we pose x1 = x;  x2 = x^2 and x3=x^3
=> P(x) = (a3)x3 + (a2)x2 + (a1)x1 + a4
We can now say that x1, x2 and x3 are FEATURES and a1,a2,a3 are PARAMETERS to a 
MULTIPLE linear regression. x1,x2,x3 are the INDEPENDENT VARIABLES! 
and P(x) = y is the Dependent variable that we are predicting.
Now we will need to model the feature we have (x) so that it will have values of:
x, x^2 and x^3, i.e the new variables.
From the same data set that we modelled a Simple and Multiple Linear Regression, we will be 
model this data set based on a polynomial regression model. It may be more accurate!
Sometimes, the trend of data is not really linear, and looks curvy. 
In this case we can use Polynomial regression methods.
To know when to use a linear model or a non-linear model, plot each independent variable
vs. the dependent variable and check the linearity visually.
Then, determine the correlation coefficient between each independent and dependent variable,
if it is >=0.7 for all cases=> There is a linear tendency and more appropriate to use Linear.
If Non-Linear regression is to be used, you can represent it as a polynomial regression, then
transform it to a multiple linear regression.
Or, you can simply execute the non-linear regression functions and classes in sklearn/linear_model
### Regression Example: Non-Linear Regression
![Non-linear regression usage](https://user-images.githubusercontent.com/98900886/202928508-01670512-af42-426d-8341-5b7acf3b4dc9.png)
![Logistic Function Formula](https://user-images.githubusercontent.com/98900886/202928524-773809d6-2132-462a-a608-03f506491253.png)


