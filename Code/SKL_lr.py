# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:22:57 2024

@author: fjanan
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import pickle
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
#start_time = time.time()

data1= pd.read_excel(r'D:/Applied Data Science/Project\Dataset_unique_103.xlsx', sheet_name='Sheet1')
## Create Unique data
df_unique=data1.drop_duplicates()
#df_unique=df_unique.drop(columns=['Variable4','Variable7'])
#df_X=df_unique.drop(columns=['Variable3', 'Variable5','Variable7','Variable8','Outcome'])
df_unique['Variable4_6']=df_unique['Variable4']*df_unique['Variable6']
df_unique['Variable4_8']=df_unique['Variable4']*df_unique['Variable8']
df_unique['Variable3_5']=df_unique['Variable3']*df_unique['Variable5']
df_unique['Variable5_6']=df_unique['Variable5']*df_unique['Variable6']
#df_unique['Variable4_7']=df_unique['Variable8']*df_unique['Variable7']
df_unique=df_unique.drop(columns=['Variable7','Variable6'])
df_X=df_unique.drop(columns=['Outcome'])
df_Y=df_unique['Outcome']
X_train, X_test, Y_train, Y_test=train_test_split(df_X, df_Y, test_size=0.2, random_state=1)
X_trainA=X_train.to_numpy()
skl_league_lr = LinearRegression()

# fit the model to the training data
skl_league_lr.fit(X_train, Y_train)

# print the estimated coefficients
print('The estimated coefficients are:')
for idx, cn in enumerate(X_train.columns):
    print(f'{cn}: {skl_league_lr.coef_[idx]:.4f}')
print(skl_league_lr.intercept_)
Y_test=pd.DataFrame(Y_test)
fig, ax = plt.subplots(1,1, figsize=(6,4))
Y_hat = skl_league_lr.predict(X_test)
ax.scatter(Y_test, Y_test['Outcome']-Y_hat, color='blue', label='Test Data')
# add text to the plot showing the R^2 value
ax.set_title(f'Test data, $R^2$ = {r2_score(Y_test, Y_hat):.2f} (Four interation vars)', fontsize=16)
ax.set_xlabel('Actual Value')
ax.set_ylabel('Residuals')
plt.show()
MSE=mean_squared_error(Y_test,Y_hat) ###0.77800
### K fold cross validation
skl_league_lr = LinearRegression()

# perform 10-fold cross-validation
r2 = cross_val_score(skl_league_lr, X_train, Y_train, cv=10, scoring='r2')
print(f'The mean R^2 value is: {r2.mean():.2f}') ## 0.59
print(f'The R^2 value standard deviation is: {r2.std():.2f}')   ## 0.08
