import pandas as pd
import numpy as np
    
def normalize2(dataframe):
    maximums=[]
    minimums=[]
    
    returnFrame = pd.DataFrame()
    
    
    atribs = dataframe.columns.values
    iter = 0
    for atrib in atribs:
        
        values = dataframe[atrib].astype('float64')
        if iter == 0:
            returnFrame = pd.DataFrame({atrib : values})
            iter += 1
        
        else:
            temp = pd.DataFrame({atrib : values})
            returnFrame = returnFrame.join(temp)
        
        
    atri = 0
    for atrib in atribs:
        print(atri)
        maximum  = np.nanmax(returnFrame[atrib])
        minimum = np.nanmin(returnFrame[atrib])
        
        values = returnFrame[atrib]
        
        
        if (maximum - minimum) != 0:
            
            linecounter = 0
            for value in values:
                
                if value != 'nan':
                    newval = value / (maximum - minimum)
                    
                    returnFrame.at[linecounter,atrib] = newval
                linecounter += 1
                
        else:
            print("min = max detected")
        
        atri += 1
    
    return returnFrame

def normalize (dataframe):
    maximums=[]
    minimums=[]
    dataframe_normalized=pd.DataFrame((np.zeros(dataframe.shape)))

    #Calculating the max and min for each attribute
    for col in range(dataframe.shape[1]):
        col_max=np.amax(dataframe.iloc[:,col])
        col_min=np.amin(dataframe.iloc[:,col])
        maximums.append(col_max)
        minimums.append(col_min)
    

    for col in range(dataframe.shape[1]): 
        col_min = minimums[col]
        col_max = maximums[col]
        for row in range (dataframe.shape[0]):
            #print('row: {}   col:{}'.format(row,col))
            if dataframe.iloc[row,col] == np.nan:
                dataframe_normalized.iloc[row,col]=np.nan
                
            else:
                dataframe_normalized.iloc[row,col]= (dataframe.iloc[row,col] - col_min) / (col_max - col_min)

    return dataframe_normalized


data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv", na_values='na')

X=data.iloc[:,1:]


#print(X)

bla = normalize2(X)

print(bla)

#X_normalized=normalize(X)

#print(X_normalized)




