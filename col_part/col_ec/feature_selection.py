import pandas as pd
import numpy as np

from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

import pickle


n_fold = 10

######################
#  DATA Generation   #
######################

green_data = pd.read_csv('../col_dataset/green.csv')

X_green = green_data.iloc[:,:62]

green_atribs = X_green.columns.values

returnFrame = pd.DataFrame(green_atribs)

a = returnFrame.join(X_green)
a.to_csv('../col_dataset/green_test.csv')



X_green = normalize(X_green, axis=0, norm='max')
#-----------------------------------------#

print (X_green)


#-----------------------------------------#
X_green = np.asarray(X_green)



for i in X_green:
    for j in i:
        if j < 0:
            print ('OLHA AQUI, SOU NEGATIVO')

Y_green = green_data['consensus']

green_labels = pd.unique(Y_green)

Y_green = np.asarray(Y_green)


chi, pval = chi2 (X_green, Y_green)

print(pval)

pvals = []
atrib_todrop = []

atri_index = 0
for atrib in green_atribs:
    if pval[atri_index] <= 0.01: # 99% confidence for discarding attributes
        atrib_todrop.append(atrib)
        atri_index += 1
    else:
        atri_index += 1

green_data = pd.read_csv('../col_dataset/green.csv')

bla = green_data.drop(atrib_todrop, axis=1)

X_green = bla.iloc[:,:62]
Y_green = bla['consensus']

scores = []
x_train = []
y_train = []
x_test = []
y_test = []

#######################
#   CLASSIFICATION    #
#######################


kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
clf = DecisionTreeClassifier()

for train_index, test_index in kf.split(X_green, Y_green):
    print('TRAIN:', train_index, 'TEST:', test_index)
    x_train, x_test = X_green[train_index], X_green[test_index]
    y_train, y_test = X_green[train_index], X_green[test_index]

    clf.fit(x_train, y_train)

    reses = clf.predict(x_test)

    confusion = confusion_matrix(y_test, reses)

    print (confusion)

    trueNeg = confusion[0][0]
    truePos = confusion[1][1]

    falseNeg = confusion[1][0]
    falsePos = confusion[0][1]

    total = trueNeg + truePos + falseNeg + falsePos
    acc = ((truePos+trueNeg)/total) * 100.0
    specificity = trueNeg/(trueNeg+falsePos)
    sensivity = truPos / (truePos + falseNeg)

    print(f'number of predictions was {total}')
    print(f'accuracy was {acc}')
    print(f'specificity rate was {specificity}')
    print(f'sensivity rate was {sensivity}')
    print("\n")