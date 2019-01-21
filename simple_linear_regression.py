# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:29:45 2019

@author: Arnab
"""
#importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
data=pd.read_csv('Salary_Data.csv')
X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values

#Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#Fitting the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the train and test results
y_pred_test=regressor.predict(X_test)
y_pred_train=regressor.predict(X_train)

#Plotting the training set results
plt.scatter(X_train,Y_train,color='Red')
plt.plot(X_train,y_pred_train,color='Blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Plotting the test set results
plt.scatter(X_test,Y_test,color='Red')
plt.plot(X_train,y_pred_train,color='Blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


