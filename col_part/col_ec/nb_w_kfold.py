import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import interp
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

from sklearn.utils import resample

n_fold = 10

##########################
# Data IO and generation #
#######################3##

# Import some data to play with

green_data = pd.read_csv('../col_dataset/green.csv')
hinselmann_data = pd.read_csv('../col_dataset/hinselmann.csv')
schiller_data = pd.read_csv('../col_dataset/schiller.csv')

green_featureNames = green_data.columns.values
hinselmann_featureNames = hinselmann_data.columns.values
schiller_featureNames = schiller_data.columns.values

print ('nr_features green:')
print (len(green_featureNames))
print ('nr_features hinselmann:')
print (len(hinselmann_featureNames))
print ('nr_features schiller:')
print (len(schiller_featureNames))

X_green = green_data.iloc[:,:62]
X_hinselmann = hinselmann_data.iloc[:,:62]
X_schiller = schiller_data.iloc[:,:62]

X_green = np.asarray(X_green)
X_hinselmann = np.asarray(X_hinselmann)
X_schiller = np.asarray(X_schiller)

Y_green = green_data['consensus']
Y_hinselmann = hinselmann_data['consensus']
Y_schiller = schiller_data['consensus']

green_labels = pd.unique(Y_green)
hinselmann_labels = pd.unique(Y_hinselmann)
schiller_labels = pd.unique(Y_schiller)

Y_green = np.asarray(Y_green)
Y_hinselmann = np.asarray(Y_hinselmann)
Y_schiller = np.asarray(Y_schiller)


kf = StratifiedKFold(n_splits = n_fold, random_state = None, shuffle = False)

scores = []
x_train = []
y_train = []
x_test = []
y_test = []

#############################################################################
# Normalization (COMMENT IT IF WANT TO CHECK RESULTS WITH NO NORMALIZATION) #
#############################################################################
#X_green = normalize(X_green, axis=0, norm='max')

###################################
# Classification and ROC Analysis #
###################################
clf = GaussianNB()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
fold = 0

####################################################################
# RESAMPLING (COMMENT IT IF WANT TO CHECK RESULTS WITH NO RESAMPLE #
####################################################################
#X_green, Y_green = resample(X_green, Y_green)

for train_index, test_index in kf.split(X_green, Y_green):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = X_green[train_index], X_green[test_index]
    y_train, y_test = Y_green[train_index], Y_green[test_index]

    # Binarize the output
    y_test_bin = label_binarize(y_test, green_labels)
    # n_classes_nb = y_test_bin_nb.shape[1]

    probas_ = clf.fit(x_train, y_train).predict_log_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test_bin, probas_[:, 0])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    reses = clf.predict(x_test)
    confusion = confusion_matrix(y_test, reses, green_labels)

    trueNeg = confusion[0][0]
    truePos = confusion[1][1]

    falseNeg = confusion[1][0]
    falsePos = confusion[0][1]

    total = trueNeg + truePos + falseNeg + falsePos
    acc = ((truePos + trueNeg) / total) * 100.0
    specificity = trueNeg / (trueNeg + falsePos)
    sensivity = truePos / (truePos + falseNeg)

    print(f"Performances for Naive Bayes at fold {fold} where")
    print(confusion)
    print(f'number of predictions was {total}')
    print(f'accuracy was {acc}')
    print(f'specificity rate was {specificity}')
    print(f'sensivity rate was {sensivity}')
    print("\n")
    fold += 1
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Random', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Chart for Naive Bayes and {n_fold} folds')
plt.legend(loc="lower right")
plt.show()
