import pandas as pd
import numpy as np
from numpy import dtype

def fill_median_by_label(labels, dataframe):
    pos_index = []
    neg_index = []
    returnFrame = pd.DataFrame()
    
    atribs = dataframe.columns.values
    
    
    index = 0
    for label in labels:
        if label == 'pos':
            pos_index.append(index)
            index += 1
        else:
            neg_index.append(index)
            index += 1
       
       
    iter = 0     
    for atrib in atribs:
        neg_vals = []
        pos_vals = []
        
        print(atrib)
        
        vals = dataframe[atrib]
        
        for index in pos_index:
            pos_vals.append(vals[index])
            
        for index in neg_index:
            neg_vals.append(vals[index])
        
        
        pos_median = np.nanmedian(pos_vals)
        neg_median = np.nanmedian(neg_vals)
        
        #print(pos_median)
        #print(neg_median)
        
        nan_index = dataframe[atrib].index[dataframe[atrib].apply(np.isnan)]
        
        print(f"attribute {atrib} with {(len(nan_index)/len(vals))*100.0} % missing values")
                
        for index in nan_index:
             
            if index in pos_index:
                vals.iloc[index] = pos_median
                 
            elif index in neg_index:
                vals.iloc[index] = neg_median
                 
            else:
                print("troubleshoot")
                 
        
        if iter == 0:
            returnFrame = pd.DataFrame({atrib : vals})
            iter+=1
            
        else:
            temp = pd.DataFrame({atrib : vals})
            returnFrame = returnFrame.join(temp)
            
        
        
    return returnFrame
    
    
    

def fill_median(dataframe):
    
    atribs = dataframe.columns.values
    returnFrame = pd.DataFrame()
    
    
    iter = 0
    for atrib in atribs:
        print(atrib)
        vals = dataframe[atrib]
        
        median = np.nanmedian(vals)
        
        print(f"median for atribute {atrib} was {median}")
        
        vals = vals.fillna(median)
        
        if iter == 0:
            returnFrame = pd.DataFrame({atrib : vals})
            iter += 1
            
        else:
            temp = pd.DataFrame({atrib : vals})
            returnFrame = returnFrame.join(temp)
        
        
    return returnFrame

     

data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv", na_values='na')

X=data.iloc[:,1:]

Y = data['class']


#x = fill_median(X)
#print(X)
#print(X)

#x = fill_median_by_label(Y,X)
#print(X)
#print(x)


#x.to_pickle('scania_bymedian.pkl')



        
        