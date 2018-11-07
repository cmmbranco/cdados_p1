from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from pymining import seqmining
from prefixspan import PrefixSpan

def columnisbinary(column):
    
    col = column.unique()
    
    for val in col:
        if val == 0 or val == 1 or val is True or val is False:
            pass
        else:
            return False
    
    return True

        
def widthnumericpreprocess(df, bins):
    #label_encoder = LabelEncoder()
    dummy_encoder = OneHotEncoder(categories='auto')
    pdf = pd.DataFrame()
    
    
    bin_map = {}
    numbs = []
    
    
    for atrib in df:
        
        
        if columnisbinary(df[atrib]):
            pdf = pd.concat([pdf, df[atrib]], axis=1)
            
        
        elif df[atrib].dtype == np.float64 or df[atrib].dtype == np.int64:
            
            #can make bins
            if df[atrib].nunique() >= bins:
                intervals, bin1 = pd.cut(df[atrib], bins, retbins=True) 
                
                vec = []
                
                counter = 0
                
                while counter < len(bin1)-1:
                    a = f"]{bin1[counter]} , {bin1[counter+1]}]"
                    vec.append([a, counter])
                    counter += 1
                    
                #store bin mapping (binval, binID)
                bin_map[atrib] = vec
                
                
                vec = df[atrib]
                
                for val in range(len(df[atrib])):
                    
                    counter = 0
                    
                    while df.iloc[val][atrib] > bin1[counter+1]:
                        counter += 1
                    
                    vec.at[val] = counter
                
                numbs = []
                
                for i in vec:
                    if i not in numbs:
                        numbs.append(int(i))
                        
                numbs = sorted(numbs, reverse=False)
                
                #print(range(len(bin1) - 2 ))
                
                # Fitting One Hot Encoding on train data
                temp = dummy_encoder.fit_transform(vec.values.reshape(-1,1)).toarray()
                
                
                
                #print(temp)
                # Changing encoded features into a dataframe with new column names
                temp = pd.DataFrame(temp,
                                    columns=[(atrib + "_" + str(i)) for i in numbs])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the data frame
            
            temp = temp.set_index(df.index.values)
            # adding the new One Hot Encoded varibales to the dataframe
            pdf = pd.concat([pdf, temp], axis=1)
                
            
        #column is not binary and cannot be discritized by formula
        else:
            print('column with n dif values less than bins or non-numeric')
            
            
    
    return pdf, bin_map

        
def printmapping(mapping):
    
    print('\n')
    print('Printing Mapping!!!')
    for x in mapping:
        print(f"atrib {x}")
        print(mapping[x])
 



data = pd.read_pickle('../../scania_pickles/train/scania_train_subsampled_split_na_normalized.pkl')

Y = data['class']
X = data.iloc[:,1:]

data_1, map = widthnumericpreprocess(X, 4)


#data = pd.to_pickle('../../scania_pickles/train/scania_train_4bin_subsampled_split_na_normalized.pkl')


# #page = widthnumericpreprocess(page, 4)
# 
# 
# 
# #print(iono)
# #print(page)
# 
print('going apri')
apri = apriori(data_1, min_support=0.6, use_colnames=True)
print(apri)
# pageapri = apriori(page, min_support=0.6, use_colnames=True)
# 
# print(ionoapri)
# 
# 
# ionorules = association_rules(ionoapri, metric="confidence", min_threshold=0.9)
# print(ionorules)
# 
# 
#     
#     
#     
#     
# 
# from pymining import assocrules
# 
# rules = assocrules.mine_assoc_rules(ionoapri, min_support=1500, min_confidence=0.8)
# rules
 
# freq_seqs = seqmining.freq_seq_enum(iono, 550)
# print(freq_seqs)
 
#seqmining
# fp = open('../../datasets/sign.txt')
# line = fp.readline()
# seqdata = []
# while line:
#     seqdata.append(line.strip().split(' '))
#     line = fp.readline()
# fp.close()

seqdata = np.asarray(data_1)
freq_seqs = seqmining.freq_seq_enum(seqdata, 550)
print("frequent items")
print(sorted(freq_seqs), '\n')
 
print("prefix span")
ps = PrefixSpan(seqdata)
print(ps.frequent(500, closed=True),'\n')
 
 
print("topk")
print(ps.topk(5, closed=True), '\n')

