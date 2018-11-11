import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_selection import chi2







train = pd.read_pickle('../../scania_pickles/train/scania_train_smoted_split_na_normalized.pkl')

test = pd.read_pickle('../../scania_pickles/test/scania_test_split_na_normalized.pkl')

x_train = train.iloc[:,1:]
y_train = train['class']

atribs = train.columns.values


chi, pval = chi2(x_train, y_train)
 
#print(pval)
 
pvals = []
atrib_todrop = []
 
 
atribs = x_train.columns.values
 
 
dic = {}
 
index = 0
for val in pval:
    dic[index] = val
    index += 1
     
 
dic = sorted(dic.items(), key=lambda kv: kv[1], reverse=False)
     
print(dic)
 
i = 0
 
to_stay = []
 
for pair in dic:
    if i == 21: #21
        break
    to_stay.append(pair[0])
    i+=1
     
i=0
 
for atrib in atribs:
    if i in to_stay:
        i += 1
    else:
        i += 1
        atrib_todrop.append(atrib)
 
x_train = x_train.drop(atrib_todrop, axis=1)



bla = test.drop(atrib_todrop, axis=1)
x_test = bla.iloc[:,1:]
y_test = bla['class']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
clf = DecisionTreeClassifier()


clf.fit(x_train,y_train)

reses = clf.predict(x_test)

confusion = confusion_matrix(y_test, reses)

print(confusion)

trueNeg = confusion[0][0]   
truePos = confusion[1][1]  
    
falseNeg = confusion[1][0]  
falsePos = confusion[0][1]  
         
total = trueNeg + truePos + falseNeg + falsePos
acc = ((truePos+trueNeg)/total) * 100.0
specificity = trueNeg / (trueNeg + falsePos)
sensivity = truePos / (truePos + falseNeg)

print(f'number of predictions was {total}')
print(f'accuracy was {acc}')
print(f'specificity rate was {specificity}')
print(f'sensivity rate was {sensivity}')
print("\n")