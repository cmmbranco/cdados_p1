import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE as smt
from sklearn.model_selection import train_test_split


def plot_2d_space(X, y, label='Classes'):
    plt.figure()
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    
    
    
data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")
data = data.replace({'na': '-1'}, regex=True)
X=data.iloc[:,1:]
X = np.asarray(X)

Y=data['class']


x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, stratify=Y)



plot_2d_space(x_train, y_train, 'UN-SMOTED')

plt.figure()
pd.value_counts(data['class']).plot.bar()
plt.title('Class nº Occur with 70-30 split, stratify on class')
plt.xlabel('Class')
plt.ylabel('Frequency')

smote = smt(ratio='minority')
X_train_smoted, Y_train_smoted = smote.fit_sample(x_train, y_train)

x_train = X_train_smoted
y_train = Y_train_smoted

plot_2d_space(X_train_smoted, Y_train_smoted, 'SMOTE over-sampling')
    
    
plt.show()
    
    