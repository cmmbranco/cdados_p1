import pandas as pd


data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")



featureNames = data.columns.values

datashape = data.shape
nrows = data.shape[0]

##extract positive entries for resampling
posframe = pd.DataFrame(columns=(featureNames))
counter = 0
pos = 0
while counter < nrows:
    row = data.iloc[counter]
    if "pos" in row[0]:
        posframe.loc[pos] = row
        pos += 1
         
    counter += 1



##print(posframe)


##
##extract feature percent missing from positives
##


dic = {}

for column in posframe:
    dic[column] = 0
    
    for x in posframe[column]:
        if x == 'na':
            dic[column] += 1
    


for word in dic:
    dic[word] = (dic[word]/posframe.shape[0])*100.0

for x in dic:
    if dic[x] >= 60:
        print(f"{x} with {dic[x]}% missing")











