import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_selection import chi2

#############
# FUNCTIONS #
#############

def columnisbinary(column):
    col = column.unique()

    for val in col:
        if val == 0 or val == 1 or val is True or val is False:
            pass
        else:
            return False

    return True


def widthnumericpreprocess(df, bins):
    # label_encoder = LabelEncoder()
    dummy_encoder = OneHotEncoder()
    pdf = pd.DataFrame()

    bin_map = {}
    numbs = []

    for atrib in df:

        if columnisbinary(df[atrib]):
            pdf = pd.concat([pdf, df[atrib]], axis=1)


        elif df[atrib].dtype == np.float64 or df[atrib].dtype == np.int64:

            # can make bins
            if df[atrib].nunique() >= bins:
                intervals, bin1 = pd.cut(df[atrib], bins, retbins=True)

                vec = []

                counter = 0

                while counter < len(bin1) - 1:
                    a = f"]{bin1[counter]} , {bin1[counter+1]}]"
                    vec.append([a, counter])
                    counter += 1

                # store bin mapping (binval, binID)
                bin_map[atrib] = vec

                vec = df[atrib]

                for val in range(len(df[atrib])):

                    counter = 0

                    while df.iloc[val][atrib] > bin1[counter + 1]:
                        counter += 1

                    vec.at[val] = counter

                numbs = []

                iter = 0
                for pair in bin1:
                    if iter == len(bin1) - 1:
                        break
                    numbs.append(iter)
                    iter += 1
                    # print(numbs)

                numbs = sorted(numbs, reverse=False)
                # print(numbs)


                bla = pd.Series(data=numbs)
                # print(vec)
                # print(bla)

                vec = vec.append(bla)
                # print(bla)


                # print(vec)
                # print(range(len(bin1) - 2 ))

                # Fitting One Hot Encoding on train data
                temp = dummy_encoder.fit_transform(vec.values.reshape(-1, 1)).toarray()

                temp = temp[:len(temp) - 4]

                # print(temp)
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


            else:
                print('debug \n')
                return
            # temp = temp.set_index(df.index.values)
            # adding the new One Hot Encoded varibales to the dataframe
            # print(pdf)

            pdf = pd.concat([pdf, temp], axis=1)

            # print(pdf)

        # column is not binary and cannot be discritized by formula
        else:
            print('column with n dif values less than bins or non-numeric')

    return pdf, bin_map

def printmapping(mapping):
    print('\n')
    print('Printing Mapping!!!')
    for x in mapping:
        print(f"atrib {x}")
        print(mapping[x])


##########################
# DATA IO AND GENERATION #
##########################

data = pd.read_csv('../col_dataset/green_normsmtdisc.csv')

X = data.iloc[:,1:187]
Y = data['consensus']

print (X)

#chi, pval = chi2(X, Y)

#pvals = []
#atrib_todrop = []

#atribs = X.columns.values

#dic = {}

#index = 0
#for val in pval:
#    dic[index] = val
#    index += 1

#dic = sorted(dic.items(), key=lambda kv: kv[1], reverse=True)

#i = 0

#to_stay = []

#for pair in dic:
#    if i == 21:
#        break
#    to_stay.append(pair[0])
#    i += 1

#i = 0

#print (to_stay)

#atribs_to_stay = []

#for atrib in atribs:
#    if i in to_stay:
#        atribs_to_stay.append(atrib)
#        i += 1
#    else:
#        i += 1
#        atrib_todrop.append(atrib)

#data_1 = X.drop(atrib_todrop, axis=1)

#print('staying features are:')

#for atrib in atribs_to_stay:
#    print (atrib)


data_1 = X

#data_1, map = widthnumericpreprocess(X, 3)

#####################
# APRIORI ALGORITHM #
#####################


    ########################################
    # Generation of nr_rules/support graph #
    ########################################

print('APRIORI ALGORITHM')
freq_items = apriori(data_1, min_support=0.60, use_colnames=True)

print('ASSOCIATION RULES')
rules = association_rules(freq_items, metric='lift', min_threshold=1.05)

print(f'Total Associtation Rules: {len(rules)}')

lifts = []
convs_index = []

after_lift = []
for row in rules.iterrows():
    if row[1].lift >= 1.05:
        after_lift.append(row)
        #print('Rows with lift > 1.05:')
        #print('LIFT')
        #print (row[1].lift)
        #print('CONVICTION')
        #print (row[1].conviction)

print(f'Found {len(after_lift)} rules with lift > 1.05')

after_conv = []

iter = 0

print('Rule Respecting Criteria')
for rule in rules.iterrows():
    if rule[1].conviction <= 1.2:
        after_conv.append(rule)

print(f'Found {len(after_conv)} rules with conviction <= 1.2 \n')

for rule in after_conv:
    print (rule)

############
# GRAPHICS #
############
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('Support')
#plt.ylabel('Nr_Rules')
#plt.title('')
#plt.show()