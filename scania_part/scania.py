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



data = pd.read_csv("../aps_dataset/aps_failure_training_set.csv")

print(data)

















