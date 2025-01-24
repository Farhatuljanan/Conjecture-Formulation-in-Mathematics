# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 08:26:00 2024

@author: farha
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

# Example dataset
data1= pd.read_excel(r'D:/Applied Data Science/Project\Dataset_unique_103.xlsx', sheet_name='Sheet1')
## Create Unique data
df=data1.drop_duplicates()


# Step 1: Correlation Matrix
print("Correlation Matrix:")
corr_matrix = df.drop(columns=['Outcome']).corr()
print(corr_matrix)

# Visualize correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Step 2: Variance Inflation Factor (VIF)
X = df.drop(columns=['Outcome'])  # Predictor variables
X['Intercept'] = 1  # Add constant for intercept calculation

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# Step 3: Condition Number
from numpy.linalg import cond

condition_number = cond(X.drop(columns=['Intercept']).values)
print("\nCondition Number:", condition_number)
