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
        
        if (maximum - minimum) == 0:
            print(f"maximum is {maximum} and minimum is {minimum}")
        
            
        linecounter = 0
        for value in values:
                
            try:
                a = str(value)
                
                    
                if a != 'nan':
                    
                    if (maximum - minimum) != 0:
                        
                        newval = value / (maximum - minimum)
                        
                    else:
                        newval = value/maximum
                    
                    returnFrame.at[linecounter,atrib] = newval
                linecounter += 1
                    
            except:
                    
                newval = value / (maximum - minimum)
                        
                returnFrame.at[linecounter,atrib] = newval
                linecounter += 1
                        
        
        atri += 1
    return returnFrame



data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv", na_values='na')

X=data.iloc[:,1:]


#print(X)

bla = normalize2(X)

print(bla)

print(X)

#X_normalized=normalize(X)

#print(X_normalized)




