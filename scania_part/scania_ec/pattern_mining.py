from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

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
    removed = []
    dummy_encoder = OneHotEncoder(categories='auto')
    pdf = pd.DataFrame()
    
    
    bin_map = {}
    numbs = []
    
    atricount = 0
    
    for atrib in df:
        
   
    
        atricount += 1
        
        if columnisbinary(df[atrib]):
            pdf = pd.concat([pdf, df[atrib]], axis=1)
            
        
        elif df[atrib].dtype == np.float64 or df[atrib].dtype == np.int64:
            
            #can make bins
            
            if df[atrib].nunique() > 2:
                #print('more unique than bins')
                intervals, bin1 = pd.cut(df[atrib], bins, retbins=True) 
                
                
                #print(intervals)
                
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
                
                iter = 0
                for pair in bin1:
                    if iter == len(bin1)-1:
                        break
                    numbs.append(iter)
                    iter +=1    
                #print(numbs)
                
                        
                numbs = sorted(numbs, reverse=False)
                #print(numbs)
                
                
                bla = pd.Series(data=numbs)
                #print(len(vec))
                #print(bla)
                
                vec = vec.append(bla)
                
                
                #print(bla)
                
                
                #print(vec)
                #print(range(len(bin1) - 2 ))
                
                # Fitting One Hot Encoding on train data
                temp = dummy_encoder.fit_transform(vec.values.reshape(-1,1)).toarray()
                              
                #print(temp)
                temp = temp[:len(temp)-4]
                
                #print(atrib)
                #print(temp)
                #print(numbs)
                
                #print(temp)
                # Changing encoded features into a dataframe with new column names
                temp = pd.DataFrame(temp,
                                    columns=[(atrib + "_" + str(i)) for i in numbs])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the data frame
            elif df[atrib].nunique() == 2:
                vals = df[atrib].unique()
                temp = df[atrib]
                
                
                
                pair = f"old: {vals[0]} new 0, old: {vals[1]} new 1"
                bin_map[atrib] = pair
                
                temp = temp.map({vals[0]: 0, vals[1]: 1})
                
            
            
            elif df[atrib].nunique() == 1:
                removed.append(atrib)
                print('removed atrib')
                
            
            else:
                print(df[atrib].nunique())
                print('debug \n')
                return
            #temp = temp.set_index(df.index.values)
            # adding the new One Hot Encoded varibales to the dataframe
            #print(pdf)
            
            
            if atrib not in removed:
                pdf = pd.concat([pdf, temp], axis=1)
                
            #print(pdf)
            
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
 



data = pd.read_pickle('../../scania_pickles/train/scania_train_bymedian.pkl')



Y = data['class']
X = data.iloc[:,1:]

 
chi, pval = chi2(X, Y)
 
#print(pval)
 
pvals = []
atrib_todrop = []
 
 
atribs = X.columns.values
 
 
dic = {}
 
index = 0
for val in pval:
    dic[index] = val
    index += 1
     
 
dic = sorted(dic.items(), key=lambda kv: kv[1], reverse=False)
     
#print(dic)
 
i = 0
 
to_stay = []
 
for pair in dic:
    if i == 21: #20 plus the nan 
        break
    to_stay.append(pair[0])
    i+=1
     
i=0
 
atribs_to_stay = []
 
for atrib in atribs:
    if i in to_stay:
        atribs_to_stay.append(atrib)
        i += 1
    else:
        i += 1
        atrib_todrop.append(atrib)
 
data = data.drop(atrib_todrop, axis=1)

data = data.sample(n = 500, axis = 0, replace=True)

data = data.reset_index()

data = data.iloc[:,1:]

Y = data['class']
X = data.iloc[:,1:]
 
#print(f"staying features are: {len(atribs_to_stay)}")

#for atrib in atribs_to_stay:
    #print(atrib)


#print(X)
#print('numeric')
data_1, map = widthnumericpreprocess(X, 4)
#print('after')
dataatri = []
# for atrib in data_1:
#     dataatri.append(atrib)

#print(len(dataatri))
#print(data_1)

support = 0.98

support_vec = []
freq_items_len_vec = []

n_rules = []
avg_lift_vec = []

print(support)
#support_vec.append(support)


#print('going apri')
#     atcount = 0
#     for atrib in data_1:
#         atcount += 1
#         
#     print(atcount)

freq_items = apriori(data_1, min_support=support, use_colnames=True)

# if support > 0.8:
#    support += 0.0250
# else:
#    support += 0.1000
   
#print(f"next support is: {support}")
#print(freq_items)

print('found frequent items:')
print(len(freq_items))

#freq_items_len_vec.append(len(freq_items))



#print('\n')
#print('associating')
rules = association_rules(freq_items, metric="confidence", min_threshold=0.95)


print(f"total association rules {len(rules)}") 

#n_rules.append(len(rules)) 

lifts = []
convs_index = []

after_lift = [] 
for row in rules.iterrows():
    lifts.append(row[1].lift)
    if row[1].lift > 1.05 or row[1].lift < 0.95:
        after_lift.append(row)


avg_lift = np.average(lifts)

#avg_lift_vec.append(avg_lift)



print(f"avg lift for rules were {avg_lift}")
after_conv = []

iter = 0

  
print("filtering by rules with lift < 0.95 or > 1.05")
print(f"found {len(after_lift)}")
for rule in after_lift:
    if rule[1].conviction <= 1.2:
        after_conv.append(rule)
      

print(f"found {len(after_conv)} rules with conviction <= 1.2 \n")

print('printing rules')
     
for rule in after_conv:
    print(rule)

#     for rule in after_conv:
#         print(rule)
# y_pos = np.arange(max(avg_lift_vec))
#   
# plt.bar(y_pos, n_rules, align='center', alpha=0.5)
# plt.xticks(y_pos, support_vec, rotation ='vertical')
# plt.ylabel('Nº Freq Items')
# plt.title('Freq. Items by support')
# y_pos = np.arange(len(support_vec))


# y_pos = np.arange(max(freq_items_len_vec))
# x_pos = np.arange(len(support_vec))
# 
# plt.bar(y_pos, freq_items_len_vec, align='center', alpha=0.5)
# plt.xticks(x_pos, support_vec, rotation ='vertical')
# plt.ylabel('Nº Freq Items')
# plt.title('Freq. Items by support')
#  
# plt.show()
