import pandas as pd
import numpy as np

from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE as smt

n_fold = 10

#####################
#    DISCRETIZER    #
#####################
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

                for i in vec:
                    if i not in numbs:
                        numbs.append(int(i))

                numbs = sorted(numbs, reverse=False)

                # print(range(len(bin1) - 2 ))

                # Fitting One Hot Encoding on train data
                temp = dummy_encoder.fit_transform(vec.values.reshape(-1, 1)).toarray()

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

######################
#  DATA Generation   #
######################

green_data = pd.read_csv('../col_dataset/green.csv')

Y_green = green_data['consensus']
Y_green = np.asarray(Y_green)
green_labels = pd.unique(Y_green)
X_green = pd.read_csv('../col_dataset/green_test.csv')
X_green = X_green.iloc[:,1:]

green_atribs = X_green.columns.values

#data_1, map = widthnumericpreprocess(X_green, 4)

#print (data_1)

#y = pd.DataFrame(data=data_1)

#y.to_csv('../col_dataset/green_test.csv')


chi, pval = chi2 (X_green, Y_green)

print(chi)
print(pval)

pvals = []
atrib_todrop = []

atri_index = 0
for atrib in green_atribs:
    if pval[atri_index] <= 0.001:  #99% confidence for discarding attributes
        print(atrib)
        atrib_todrop.append(atrib)
        atri_index += 1
    else:
        atri_index += 1

print (atrib_todrop)

green_data = pd.read_csv('../col_dataset/green_test.csv')

bla = green_data.drop(atrib_todrop, axis=1)

X_green = bla.iloc[:,1:]
X_green = np.asarray(X_green)

scores = []
x_train = []
y_train = []
x_test = []
y_test = []

#################
# PREPROCESSING #
#################
# Normalization (comment it if want to check results with no normalization)
X_green = normalize(X_green, axis=0, norm='max')

# Resampling (comment it if want to check results with no resampling)
#X_hinselmann, Y_hinselmann = resample(X_hinselmann, Y_hinselmann)

# Smote (comment it if want to check results with no smote)
smote = smt(ratio='minority')
X_green, Y_green = smote.fit_sample(X_green, Y_green)

#######################
#   CLASSIFICATION    #
#######################

kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
clf = KNeighborsClassifier(n_neighbors=3)

for train_index, test_index in kf.split(X_green, Y_green):
    #print('TRAIN:', train_index, 'TEST:', test_index)
    x_train, x_test = X_green[train_index], X_green[test_index]
    y_train, y_test = Y_green[train_index], Y_green[test_index]

    clf.fit(x_train, y_train)

    reses = clf.predict(x_test)

    confusion = confusion_matrix(y_test, reses)

    print (confusion)

    trueNeg = confusion[0][0]
    truePos = confusion[1][1]

    falseNeg = confusion[1][0]
    falsePos = confusion[0][1]

    total = trueNeg + truePos + falseNeg + falsePos
    acc = ((truePos+trueNeg)/total) * 100.0
    specificity = trueNeg/(trueNeg+falsePos)
    sensivity = truePos / (truePos + falseNeg)

    print(f'number of predictions was {total}')
    print(f'accuracy was {acc}')
    print(f'specificity rate was {specificity}')
    print(f'sensivity rate was {sensivity}')
    print("\n")