import pandas as pd
import numpy as np


def split_na(dataframe):
    
    returnFrame = pd.DataFrame()
    
    atribs = dataframe.columns.values
    
    atri = 0
    iter = 0
    for atrib in atribs:
        print(atri)
        
        values = dataframe[atrib]
        
        atrib_na = atrib + '_na'
        nas = values
        
        values = values.fillna(0)
        
        index = 0
        for na in nas:
            try:
                a = str(na)
                
                if a == 'nan':
                    nas.iloc[index] = 1
                    
                else:
                    nas.iloc[index] = 0
                    
                
                index +=1
            
            except:
                nas.iloc[index] = 0
                index += 1
        
        print(index)
        
        if iter == 0:
        
            returnFrame = pd.DataFrame({atrib : values})
            temp = pd.DataFrame({atrib_na : nas})
            returnFrame = returnFrame.join(temp)
            iter += 1
            
        
        else:
            temp = pd.DataFrame({atrib : values})
            temp2 = pd.DataFrame({atrib_na : nas})
            returnFrame = returnFrame.join(temp)
            returnFrame = returnFrame.join(temp2)
        
        
    
        
        atri += 1
        print(returnFrame)
    
    return returnFrame


data = pd.read_csv("../scania_dataset/aps_failure_test_set.csv", na_values='na')

X=data.iloc[:,1:]

Y = data['class']

returnFrame = pd.DataFrame({'class' : Y})

x = split_na(X)

a = returnFrame.join(x)

#a.to_pickle('aps_test_split_na.pkl')
