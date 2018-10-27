import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats



data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")

Y = data['class']

data = data.iloc[:,1:]

#print(Y)

#data = np.asarray(data)

atribRem = {}

atribs = data.columns.values

for atrib in atribs:
    #print(atrib)
    values = data[atrib]
    tval = []
    for value in values:
        if value != 'na':
            tval.append(float(value))
    
    n_samp = len(tval)
    

    q75, q25 = np.percentile(tval, [75 ,25])
    #print(f"quartile 75 {q75}")
    iqr = q75 - q25
    
    #print(f"iqr for attribute {atrib} was {iqr}")
    
    
    minThresh = q25 - 1.5*iqr
    maxThresh = q75 + 1.5*iqr
    
    atribRem[atrib] = (minThresh, maxThresh, iqr)




outlier_lines = []


for atrib in atribs:    
    lineCounter = 0
    
    #print(atrib)
    
    
    values = data[atrib]
    for value in values:
        if value == 'na':
            lineCounter += 1
        
        
        else:
            if (float(value) < atribRem[atrib][0] or float(value) > atribRem[atrib][1]) and atribRem[atrib][2] != 0:
                #print("outlier")
                outlier_lines.append(lineCounter)
            
            lineCounter += 1

outlier_lines = np.unique(outlier_lines)


#print(outlier_lines)

b = sorted(outlier_lines, reverse=True)


print(len(b))

"""
x = data
#print(x)

x1 = x.drop(b)
print(x1)
# 


Y = Y.drop(b)
#print(Y)


negcount = 0
counter = 0
for x in Y:
    counter+=1
    if x == 'neg':
        negcount +=1

pos = counter - negcount


print(f"total {counter} wiht {pos} pos and {negcount}")
"""

"""
##distribuição mantém apos remover os outliers vamos remover os na.


vec = []
for tuple in x1.itertuples(index=True):
    
    for x in tuple:
        if x == 'na':
            vec.append(tuple[0])
            break



vec = pd.unique(vec)

b = sorted(vec, reverse=True)

x2 = x1.drop(b)

#print(x2)


#ficam 0 entradas ou seja todas têm pelo menos 1 na"""


