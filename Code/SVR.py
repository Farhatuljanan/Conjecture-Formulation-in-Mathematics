# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:40:47 2024

@author: farha
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

data1= pd.read_excel(r'D:/Applied Data Science/Project\Dataset_unique_103.xlsx', sheet_name='Sheet1')
## Create Unique data
df_unique=data1.drop_duplicates()
df_X=df_unique.drop(columns=['Outcome'])
df_Y=df_unique['Outcome']
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=42)

# Feature scaling (important for SVR)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).flatten()


svr = SVR()

# Define parameter grid
#param_grid = {
 #   'C': [0.1, 1, 10, 100],  # Regularization parameter
    #'epsilon': [0.01, 0.1, 0.5, 1],  # Margin tolerance
  #  'kernel': ['linear', 'rbf', 'poly'],  # Kernel types
    #'gamma': ['scale', 'auto', 0.1, 1]  # Kernel coefficient
#}
# Train an SVR model
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)  # RBF kernel, default parameters
svr.fit(X_train_scaled, y_train_scaled)
#grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
#grid_search.fit(X_train_scaled, y_train_scaled)

# Best parameters and performance
#print("Best Parameters:", grid_search.best_params_)
#print("Best Score:", -grid_search.best_score_)

# Evaluate on test set
#best_model = grid_search.best_estimator_
#y_pred_scaled = best_model.predict(X_test)
# Make predictions
y_pred_scaled = svr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
y_Test=pd.DataFrame(y_test)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.scatter(y_Test, y_Test-y_pred, color='blue', label='Test Data')
# add text to the plot showing the R^2 value
ax.set_title(f'Test data, $R^2$ = {r2_score(y_test, y_pred):.2f} (SVR)', fontsize=16)
ax.set_xlabel('Actual Value')
ax.set_ylabel('Residuals')
plt.show()
# Visualize results
r2 = cross_val_score(svr, X_train_scaled, y_train_scaled, cv=10, scoring='r2')
print(f'The mean R^2 value is: {r2.mean():.2f}') ## 0.59
print(f'The R^2 value standard deviation is: {r2.std():.2f}')   ## 0.08
