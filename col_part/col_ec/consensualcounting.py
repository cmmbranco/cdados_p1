import pandas as pd

green_data = pd.read_csv("../col_dataset/green.csv")
hinselmann_data = pd.read_csv("../col_dataset/hinselmann.csv")
schiller_data = pd.read_csv("../col_dataset/schiller.csv")

dic = {}

dic['green'] = green_data.shape[0]
dic['hinselmann'] = hinselmann_data.shape[0]
dic['schiller'] = schiller_data.shape[0]

data_array = [green_data, hinselmann_data, schiller_data]
dic_array = ['green', 'hinselmann', 'schiller']
counter = 0

for data in data_array:
    working_data = data_array[counter].iloc[:, -1] #Last column of the data frame (CONSENSUS)

    nrows = dic[dic_array[counter]]
    i = 0
    consensus = 0
    while i < nrows:
        value = working_data.iloc[i]
        if value == 1.0:
            consensus += 1
        i += 1

    consensus_perc = (consensus/nrows)*100.0
    print (f'Consensus on {dic_array[counter]} dataset is {consensus_perc} %')
    counter += 1