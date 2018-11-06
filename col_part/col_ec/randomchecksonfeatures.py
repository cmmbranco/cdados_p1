import pandas as pd

########################################
# Checks the amount of 0's per feature #
########################################

green_data = pd.read_csv("../col_dataset/green.csv")
hinselmann_data = pd.read_csv("../col_dataset/hinselmann.csv")
schiller_data = pd.read_csv("../col_dataset/schiller.csv")

dic = {}

nrows = schiller_data.shape[0]
print ('Nrows: %d' % nrows)

for column in schiller_data:
    dic[column] = 0

    for x in green_data[column]:
        if x == 0.0:
            dic[column] += 1

for word in dic:
    dic[word] = (dic[word]/nrows)*100.0

for x in dic:
    print (f'{x} with {dic[x]}% of 0s')
