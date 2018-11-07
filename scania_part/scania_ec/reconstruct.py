import pandas as pd
import numpy as np

#data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv", na_values='na')
# 
# 

#Y = data['class']
# 
X = pd.read_pickle('../../scania_pickles/scania_train_split_na_normalized.pkl')

#Z = pd.read_pickle('../../scania_pickles/scania_train_split_na.pkl')
print('bla')
print(X.iloc[45173])
 
#print(Z)

#returnFrame = pd.DataFrame({'class' : Y})
#  
#a = returnFrame.join(X)
#  
#print(a)
#a.to_pickle('../../scania_pickles/scania_train_split_na_normalized.pkl')
