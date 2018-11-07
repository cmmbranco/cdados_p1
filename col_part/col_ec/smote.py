import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE as smt
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics.classification import accuracy_score
from matplotlib.pyplot import plot
import random


def sampler(dataframe):
    atribs = dataframe.columns.values

    print ('atribs')
    print (atribs)

    posLines = []
    negLines = []

    linecounter = 0
    neg_counter = 0
    pos_counter = 0

    for line in dataframe['consensus']:
        if line == 1.0:
            print("pos detected")
            posLines.append(dataframe.iloc[linecounter])
            pos_counter += 1
            linecounter += 1
        else:
            print("neg detected")
            negLines.append(dataframe.iloc[linecounter])
            neg_counter +=1
            linecounter += 1

    print(linecounter)

    random.shuffle(negLines)

    pos = []
    lineCounter = 0
    while lineCounter < neg_counter:
        pos.append(posLines[lineCounter])
        lineCounter += 1

    finaldata = negLines + pos

    random.shuffle(finaldata)

    newframe = pd.DataFrame(data=finaldata, columns=atribs)

    return newframe



    # print(atrib)


#    values = data[atrib]
#    for value in values:


def plot_2d_space(X, y, label='Classes'):
    plt.figure()
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')


schiller_data = pd.read_csv('../col_dataset/hinselmann.csv')
subsampled_data = sampler(schiller_data)

print (subsampled_data)

X_subsample = subsampled_data.iloc[:, :62]
X_subsample = np.asarray(X_subsample)
Y_subsample = subsampled_data['consensus']

X = schiller_data.iloc[:, :62]
X = np.asarray(X)

Y = schiller_data['consensus']

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, stratify=Y)

schiller_data['consensus'].value_counts().plot(kind='bar')

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
results = clf.predict(x_test)

acc = accuracy_score(y_test, results)

print(f"unsmoted acc was {acc}")

plot_2d_space(x_train, y_train, 'unsmoted sample')

smote = smt(ratio='minority')
X_train_smoted, Y_train_smoted = smote.fit_sample(x_train, y_train)

x_train = X_train_smoted
y_train = Y_train_smoted

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
results = clf.predict(x_test)

acc = accuracy_score(y_test, results)

print(f"smoted acc was {acc}")

plot_2d_space(X_train_smoted, Y_train_smoted, 'SMOTE over-sampling')

# resampling

from sklearn.utils import resample

x_train_res, y_train_res = resample(x_train, y_train)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train_res,y_train_res)
results = clf.predict(x_test)

acc = accuracy_score(y_test, results)
print (f'resampled acc was {acc}')

plot_2d_space(x_train_res, y_train_res, 'Resample')

plt.show()