import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE as smt
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics.classification import accuracy_score
from matplotlib.pyplot import plot
import random


def resampler(dataframe):
    
    atribs = dataframe.columns.values
    
    
    
    posLines = []
    negLines = []
    
    linecounter = 0
    neg_counter = 0
    pos_counter = 0
    
    for line in dataframe['class']:
        if line == 'pos':
            #print("pos detected")
            posLines.append(dataframe.iloc[linecounter])
            pos_counter += 1
            linecounter += 1
        else:
            #print("neg detected")
            negLines.append(dataframe.iloc[linecounter])
            linecounter += 1
    
    print(linecounter)
    
    
    random.shuffle(negLines)
   
    
    negs = []
    lineCounter = 0
    while lineCounter < pos_counter:
        negs.append(negLines[lineCounter])
        lineCounter += 1
        
    
    finaldata = posLines + negs
    
    random.shuffle(finaldata)
    
    
    newframe = pd.DataFrame(data=finaldata,columns=atribs)
    
    
    return newframe
        
    
    
    #print(atrib)
    
    
#    values = data[atrib]
#    for value in values:
        

# def plot_2d_space(X, y, label='Classes'):
#     plt.figure()
#     colors = ['#1F77B4', '#FF7F0E']
#     markers = ['o', 's']
#     for l, c, m in zip(np.unique(y), colors, markers):
#         plt.scatter(
#             X[y==l, 0],
#             X[y==l, 1],
#             c=c, label=l, marker=m
#         )
#     plt.title(label)
#     plt.legend(loc='upper right')
    

#from sklearn.utils import resample
data = pd.read_pickle('../../scania_pickles/train/scania_train_smoted_split_na_normalized.pkl')
#subsampled_data = sampler(data)
print(data)

#atribs = []




#x_train = data.iloc[:,1:]
#y_train = data['class']


#for atrib in x_train:
#    atribs.append(atrib)

#a.to_pickle('../../scania_pickles/scania_train_subsampled_split_na_normalized.pkl')


#x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, stratify=Y)


#data['class'].value_counts().plot(kind='bar')



#plot_2d_space(x_train, y_train, 'unsmoted sample')

#smote = smt(ratio='minority')

#print(smote)
#X_train_smoted, Y_train_smoted = smote.fit_sample(x_train, y_train)

#y = pd.DataFrame({'class' : Y_train_smoted})

#returnFrame = pd.DataFrame(columns=atribs, data=X_train_smoted)

#a = y.join(returnFrame)

#print(a)

#a.to_pickle('../../scania_pickles/train/scania_train_smoted_split_na_normalized.pkl')

#a['class'].value_counts().plot(kind='bar')

#plot_2d_space(X_train_smoted, Y_train_smoted, 'SMOTE over-sampling')

#plot_2d_space(x_train_res, y_train_res, 'Resample')

#plt.show()
    
    