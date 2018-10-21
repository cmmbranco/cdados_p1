import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def preprocessData(df):
    label_encoder = LabelEncoder()
    dummy_encoder = OneHotEncoder()
    pdf = pd.DataFrame()
    for att in df.columns:
        if df[att].dtype == np.float64 or df[att].dtype == np.int64:
            pdf = pd.concat([pdf, df[att]], axis=1)
        else:
            df[att] = label_encoder.fit_transform(df[att])
            # Fitting One Hot Encoding on train data
            temp = dummy_encoder.fit_transform(df[att].values.reshape(-1,1)).toarray()
            # Changing encoded features into a dataframe with new column names
            temp = pd.DataFrame(temp,
                                columns=[(att + "_" + str(i)) for i in df[att].value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the data frame
            temp = temp.set_index(df.index.values)
            # adding the new One Hot Encoded varibales to the dataframe
            pdf = pd.concat([pdf, temp], axis=1)
    
    
    
    
    copy = pdf
    dict = {}
    
    
    for word in pdf.columns:
        split = word.split('_')
        attri = ''
        for term in range(len(split)-1):
            attri = attri + split[term] + '_'
        
        if attri in dict:
           dict[attri] += 1
           
        else:
            dict[attri] = 1
                
    
    for attrib in dict:
        if dict[attrib] == 2:
            rem =  attrib + '1'
            print(rem)
            copy = copy.drop([rem], axis=1)
            
    print(copy)
        
    return copy





####
####Main Loop
####



data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")


featureNames = data.columns.values

classes = data['class']
nclasses = pd.unique(classes)
#print(classes)


totalentries = len(classes)
print(totalentries)

dic = {}


##calculate % of na per feature
for column in data:
    dic[column] = 0
    
    for x in data[column]:
        if x == 'na':
            dic[column] += 1
    

for word in dic:
    dic[word] = (dic[word]/totalentries)*100.0

for x in dic:
    if dic[x] >= 60:
        print(f"{x} with {dic[x]}% missing")
        



##extract positive entries for resampling
posframe = pd.DataFrame(columns=(featureNames))
counter = 0
pos = 0
while counter < totalentries:
    row = data.iloc[counter]
    if "pos" in row[0]:
        posframe.loc[pos] = row
        pos += 1
         
    counter += 1


##detect features with na

counter = 0
postlen = len(posframe['class'])

while counter < postlen:
    print(posframe.iloc[counter])
    counter += 1


print(posframe)
        

# sorted_by_value = sorted(dic.items(), key=lambda kv: kv[1])
# 
# #print(sorted_by_value)
# 
# for entry in sorted_by_value:
#     print(entry)


    
# for word in dic:
#     print(f"line {word} with {dic[word]} na")
    





# while counter < totalentries:
#     dic[counter] = 0
#     bla = data.iloc[counter]
#     #print(bla)
#     for x in bla:
#         if x == 'na':
#             dic[counter] += 1
#          
#     counter += 1
    


#print(sort)
