import pandas as pd
import numpy as np

data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv", na_values='na')

X=data.iloc[:,1:]

maximums=[]
minimums=[]
X_normalized=np.zeros(X.shape)

#Calculating the max and min for each attribute
for col in range(X.shape[1]):
    col_max=np.amax(X.iloc[:,col])
    col_min=np.amin(X.iloc[:,col])
    maximums.append(col_max)
    minimums.append(col_min)
    

for col in range(2): #X.shape[1]
    for row in range (X.shape[0]):
        print('row: {}   col:{}'.format(row,col))
        if X.iloc[row,col] == np.nan:
            X_normalized[row][col]=np.nan
            
        else:
            X_normalized[row][col]= (X.iloc[row,col] - minimums[col]) / (maximums[col] - minimums[col])
