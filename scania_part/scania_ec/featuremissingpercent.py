import pandas as pd



###
###Main LOOP
###


data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")


featureNames = data.columns.values

datashape = data.shape
nrows = data.shape[0]

#print(rows)

###
###calculate % of na per feature
###

dic = {}

for column in data:
    dic[column] = 0
    
    for x in data[column]:
        if x == 'na':
            dic[column] += 1


for word in dic:
    dic[word] = (dic[word]/nrows)*100.0

for x in dic:
    if dic[x] >= 60:
        print(f"{x} with {dic[x]}% missing")

