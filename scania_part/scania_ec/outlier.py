import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")

data = data.iloc[:,1:]

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
    iqr = q75 - q25
    
    minThresh = q25 - 1.5*iqr
    maxThresh = q75 + 1.5*iqr
    
    atribRem[atrib] = (minThresh, maxThresh)




outlier_lines = []
for atrib in atribs:    
    lineCounter = 0
    
    #print(atrib)
    
    values = data[atrib]
    for value in values:
        if value == 'na':
            lineCounter += 1
        
        
        else:
            if float(value) < atribRem[atrib][0] or float(value) > atribRem[atrib][1]:
                #print("outlier")
                outlier_lines.append(lineCounter)
                lineCounter += 1
    


outlier_lines = np.unique(outlier_lines)
print(outlier_lines)
