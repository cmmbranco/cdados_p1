import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

#data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv", na_values='na')

data = pd.read_pickle('../../scania_pickles/scania_test_split_na.pkl')

X=data.iloc[:,1:]
Y=data['class']
print(X)
y = pd.DataFrame({'class' : Y})


X_normalized = normalize(X, axis=0, norm='max')



atribs = []
dic = {}
a = np.transpose(X_normalized)


for atrib in X:
    atribs.append(atrib)
    
counter = 0
for atrib in atribs:
    dic[atrib] = a[counter]
    counter += 1
          




     
 
returnFrame = pd.DataFrame(dic)
#print(returnFrame)
a = y.join(returnFrame)

print(a)

# a.to_pickle('../../scania_pickles/scania_test_split_na_normalized.pkl')
