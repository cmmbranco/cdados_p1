import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE as smt
from sklearn.preprocessing import OneHotEncoder

import difflib

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
                #print(atribs)
                #print(intervals)
                #print(bin1)
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
                    if iter == len(bin1)-1:
                        break
                    numbs.append(iter)
                    iter += 1
                #print(numbs)

                #for i in range(bins):
                #    numbs.append(i)

                #for i in vec:
                #    if i not in numbs:
                #        numbs.append(int(i))

                numbs = sorted(numbs, reverse=False)

                bla = pd.Series(data=numbs)

                vec = vec.append(bla)


                # print(range(len(bin1) - 2 ))

                # Fitting One Hot Encoding on train data
                temp = dummy_encoder.fit_transform(vec.values.reshape(-1, 1)).toarray()

                temp = temp[:len(temp)-4]

                # print(temp)
                # Changing encoded features into a dataframe with new column names
                temp = pd.DataFrame(temp, columns=[(atrib + "_" + str(i)) for i in numbs])
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

##############
# DATA ENTRY #
##############

data = pd.read_csv('../col_dataset/green.csv')

X = data.iloc[:,:62]
atribs = X.columns.values
X = np.asarray(X)
Y = data['consensus']
Y = np.asarray(Y)
labels = pd.unique(Y)

#################
# NORMALIZATION #
#################
X = normalize(X, axis=0, norm='max')

#########
# SMOTE #
#########
smote = smt(ratio='minority')
X, Y = smote.fit_sample(X, Y)

a = pd.DataFrame(data=X, columns=atribs)

##############
# DISCRETIZE #
##############
X = a.iloc[:,:62]
data_1, map = widthnumericpreprocess(X, 3)

##########
# SAVING #
##########
# Working on removing parts
atrib = []
atrib.append('consensus')
data_2 = pd.DataFrame(Y, columns=atrib)

data_1 = data_1.join(data_2)

print(data_1)

data_1.to_csv('../col_dataset/green_normsmtdisc.csv')

