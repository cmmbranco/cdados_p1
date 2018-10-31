import pandas as pd

#Main Loop

#green_data = pd.read_csv("../col_dataset/green.csv")
hinselmann_data = pd.read_csv("../col_dataset/hinselmann.csv")
#schiller_data = pd.read_csv("../col_dataset/schiller.csv")

dic = {}

nrows = hinselmann_data.shape[0]
print ('Nrows: %d' % nrows)

for column in hinselmann_data:
    dic[column] = 0

    for x in hinselmann_data[column]:
        if x == 'na':
            dic[column] += 1

for word in dic:
    dic[word] = (dic[word]/nrows)*100.0

for x in dic:
    print (f'{x} with {dic[x]}% missing')
