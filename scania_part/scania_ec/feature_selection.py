import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_selection import chi2




data = pd.read_pickle('../../scania_pickles/train/scania_train_split_na_normalized.pkl')

print(data)

X = data.iloc[:,1:]
Y=data['class']


atribs = X.columns.values

chi, pval = chi2(X, Y)

print(pval)

pvals = []
atrib_todrop = []




atri_index = 0
for atrib in atribs:
    if pval[atri_index] <= 0.01: #99% confidence for discarding attribute
        atrib_todrop.append(atrib)
        atri_index += 1
    
    else:
        atri_index += 1



train = pd.read_pickle('../../scania_pickles/train/scania_train_smoted_split_na_normalized.pkl')

test = pd.read_pickle('../../scania_pickles/test/scania_test_split_na_normalized.pkl')



bla = train.drop(atrib_todrop, axis=1)

x_train = bla.iloc[:,1:]
y_train = bla['class']

bla = test.drop(atrib_todrop, axis=1)
print(bla)
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












