import pandas as pd
import numpy as np


def normalize (dataframe):
    maximums=[]
    minimums=[]
    dataframe_normalized=pd.DataFrame((np.zeros(dataframe.shape)))

    #Calculating the max and min for each attribute
    for col in range(dataframe.shape[1]):
        col_max=np.amax(dataframe.iloc[:,col])
        col_min=np.amin(dataframe.iloc[:,col])
        maximums.append(col_max)
        minimums.append(col_min)
    

    for col in range(dataframe.shape[1]): 
        col_min = minimums[col]
        col_max = maximums[col]
        for row in range (dataframe.shape[0]):
            print('row: {}   col:{}'.format(row,col))
            if dataframe.iloc[row,col] == np.nan:
                dataframe_normalized.iloc[row,col]=np.nan
                
            else:
                dataframe_normalized.iloc[row,col]= (dataframe.iloc[row,col] - col_min) / (col_max - col_min)

    return dataframe_normalized


data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv", na_values='na')

X=data.iloc[:,1:]

X_normalized=normalize(X)