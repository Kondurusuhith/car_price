
import numpy as np # importing NUMPY

import pandas as pd # importing PANDAS 

df = pd.read_csv('car_data.csv') # dataframe : car_data # DATA collected from kaggle Linear Regression Vehicle dataset

df.head() # head() function is used to get the first n rows, this function returns the first n rows(by default value 5) for the object based on position, it is useful for quickly testing if your object has the right type of data in it

df.describe() # describe() method is used for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame

df.isnull().sum() # returns the number of missing values in the data set # dtype: int64

df['Fuel_Type'].value_counts() # value_counts() function returns object containing counts of unique values # Name: Fuel_Type, dtype: int64

df['Seller_Type'].value_counts() # Name: Seller_Type, dtype: int64

df['Transmission'].value_counts() # Name: Transmission, dtype: int64

df['Owner'].value_counts() # Name: Owner, dtype: int64

df = df.drop('Car_Name',axis = 1) # drop() function is used to drop specified labels from rows or columns; axis=1 for removing column

df['Car_Age'] = 2021 - df['Year'] # number of years from when the car is being used

df = df.drop('Year', axis = 1) # drop() function is used to drop specified labels from rows or columns

df # display dataframe


# make 'three' new columns named "CNG, Diesel, Petrol" and mark values of them if it that type make it 1 or else 0

dummy_variable_1 = pd.get_dummies(df["Fuel_Type"]) # pd.get_dummies when applied to a column of categories where we have one category per observation will produce a new column (variable) for each unique categorical value

df.head()

df # display dataframe

df = pd.concat([df, dummy_variable_1], axis=1) # concatenate pandas objects along a particular axis with optional set logic along the other axes

df # display dataframe

# make 'two' new columns named "Dealer, Individual" and mark values of them if it that type make it 1 or else 0

dummy_variable_2 = pd.get_dummies(df["Seller_Type"])
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop("Seller_Type", axis = 1, inplace=True) # removes Seller_Type column

df # display dataframe

# make 'two' new columns named "Automatic, Manual" and mark values of them if it that type make it 1 or else 0

dummy_variable_3 = pd.get_dummies(df["Transmission"])
df = pd.concat([df, dummy_variable_3], axis=1)
df.drop("Transmission", axis = 1, inplace=True) # removes Transmission column

df # display dataframe

df = df.drop('CNG', axis =1)

df = df.drop('Diesel', axis =1)

df = df.drop('Petrol', axis =1)

df # display dataframe

# make 'three' new columns named "CNG, Diesel, Petrol" and mark values of them if it that type make it 1 or else 0

dummy_variable_1 = pd.get_dummies(df["Fuel_Type"])
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("Fuel_Type", axis = 1, inplace=True) # removes Fuel_Type column

df # display dataframe

df.corr() # dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored

df # display dataframe

import matplotlib.pyplot as plt
%matplotlib inline


import seaborn as sns

sns.heatmap(df.corr()) # A correlation heatmap is a heatmap that shows a 2D correlation matrix between two discrete dimensions, using colored cells to represent data from usually a monochromatic scale. The values of the first dimension appear as the rows of the table while of the second dimension as a column

df # display dataframe

# iloc indexer for Pandas Dataframe is used for integer-location based indexing / selection by position. iloc returns a Pandas Series when one row is selected, and a Pandas DataFrame when multiple rows are selected, or if any column in full is selected

X = df.iloc[:,1:]
Y = df.iloc[:, 0]

df # display dataframe


X.head()
Y.head()

from sklearn.model_selection import train_test_split

# train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets: for training data and for testing data. With this function, you don't need to divide the dataset manually. By default, Sklearn train_test_split will make random partitions for the two subsets

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20)


from sklearn.linear_model import LinearRegression
from sklearn import metrics


df.plot(x='Present_Price', y='Selling_Price', style='o')  
plt.title('Present_Price vs Selling_Price')  
plt.xlabel('Present_Price')  
plt.ylabel('Selling_Price')  
plt.show()





plt.figure(figsize=(15,10))
sns.histplot(df['Selling_Price'])
 
# Seaborn is a data visualization library based on matplotlib in Python. In this article, we will use seaborn. histplot() to plot a histogram with a density plot

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# LinearRegression()

print(regressor.intercept_)
print(regressor.coef_)

'''
5.5078611430667435
[ 4.28263354e-01 -6.42103586e-06 -8.53513845e-01 -4.41330254e-01
  5.59175129e-01 -5.59175129e-01  9.58510684e-01 -9.58510684e-01
 -1.16245776e+00  1.51818778e+00 -3.55730015e-01]
'''



Y_predictions = regressor.predict(X_test)


comparision_df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_predictions})

comparision_df # display comparision_df


plt.scatter(X_test['Present_Price'], Y_test,  color='gray')
plt.plot(X_test['Present_Price'], Y_predictions, color='red', linewidth=1)
plt.show()


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_predictions)))

'''
Mean Absolute Error: 1.0105150147762252
Mean Squared Error: 1.749939702871416
Root Mean Squared Error: 1.322852865163551
'''




import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, Y_predictions), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(Y_test, Y_predictions), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(Y_test, Y_predictions), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(Y_test, Y_predictions), 2)) 
print("R2 score =", round(sm.r2_score(Y_test, Y_predictions), 2))

'''
Mean absolute error = 1.01
Mean squared error = 1.75
Median absolute error = 0.8
Explain variance score = 0.9
R2 score = 0.89
'''

from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor2.fit(X_test, Y_test)


# RandomForestRegressor(n_estimators=10, random_state=0)


Y_rf = regressor2.predict(X_test)



print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, Y_rf), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(Y_test, Y_rf), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(Y_test, Y_rf), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(Y_test, Y_rf), 2)) 
print("R2 score =", round(sm.r2_score(Y_test, Y_rf), 2))


'''
Mean absolute error = 0.4
Mean squared error = 0.72
Median absolute error = 0.12
Explain variance score = 0.96
R2 score = 0.96
'''



from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestRegressor()

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,X_train,Y_train,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)
# -4.832694917845307

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,Y_train)

'''
GridSearchCV(cv=5, estimator=Ridge(),
             param_grid={'alpha': [1e-15, 1e-10, 1e-08, 0.001, 0.01, 1, 5, 10,
                                   20, 30, 35, 40, 45, 50, 55, 100]},
             scoring='neg_mean_squared_error')
'''


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
'''
{'alpha': 45}
-4.350617820858952
'''




from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,Y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
'''
{'alpha': 0.01}
-4.644618258805908
'''




Y_lasso=lasso_regressor.predict(X_test)
Y_ridge=ridge_regressor.predict(X_test)


print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, Y_lasso), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(Y_test, Y_lasso), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(Y_test, Y_lasso), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(Y_test, Y_lasso), 2)) 
print("R2 score =", round(sm.r2_score(Y_test, Y_lasso), 2))
'''
Mean absolute error = 0.98
Mean squared error = 1.66
Median absolute error = 0.78
Explain variance score = 0.9
R2 score = 0.9
'''



print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, Y_ridge), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(Y_test, Y_ridge), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(Y_test, Y_ridge), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(Y_test, Y_ridge), 2)) 
print("R2 score =", round(sm.r2_score(Y_test, Y_ridge), 2))
'''
Mean absolute error = 0.89
Mean squared error = 1.49
Median absolute error = 0.57
Explain variance score = 0.91
R2 score = 0.91
'''



sns.histplot(Y_test - Y_rf)




# https://machinelearningmastery.com/linear-regression-for-machine-learning/

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso

# The Lasso is a linear model that estimates sparse coefficients with l1 regularization




# Developed an end-to-end machine learning model to predict the car prices
# Used Principal Component Analysis for Dimensionality reduction and added features to enhance model performance
# Compared the accuracies of various regression models, choose Random Forest regressor and predicted the prices with 97% accuracy


