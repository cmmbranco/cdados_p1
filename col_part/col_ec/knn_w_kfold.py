import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


n_fold = 10
k = 21

##########################
# Data IO and generation #
##########################

# Import some data to play with

data = pd.read_csv('../col_dataset/schiller_normsmt.csv') #Change name to test other dataset

X = data.iloc[:,1:62]
print(X)
X = np.asarray(X)
Y = data['consensus']
labels = pd.unique(Y)
Y = np.asarray(Y)

##############
#            #
##############

kf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=False)

scores = []
x_train = []
y_train = []
x_test = []
y_test = []

###################################
# CLASSIFICATION AND ROC ANALYSIS #
###################################
clf = KNeighborsClassifier(n_neighbors=k)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
fold = 0

sens = []
spec = []
accu = []

for train_index, test_index in kf.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Binarize the output
    y_test_bin = label_binarize(y_test, labels)
    # n_classes_nb = y_test_bin_nb.shape[1]

    probas_ = clf.fit(x_train, y_train).predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test_bin, probas_[:, 0])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    reses = clf.predict(x_test)
    confusion = confusion_matrix(y_test, reses, labels)

    trueNeg = confusion[0][0]
    truePos = confusion[1][1]

    falseNeg = confusion[1][0]
    falsePos = confusion[0][1]

    total = trueNeg + truePos + falseNeg + falsePos
    acc = ((truePos + trueNeg) / total) * 100.0
    accu.append(acc)
    specificity = trueNeg / (trueNeg + falsePos)
    spec.append(specificity)
    sensivity = truePos / (truePos + falseNeg)
    sens.append(sensivity)

    print(f"Performances for KNN with k ={k} at fold {fold}")
    print(confusion)
    print(f'number of predictions was {total}')
    print(f'accuracy was {acc}')
    print(f'specificity rate was {specificity}')
    print(f'sensivity rate was {sensivity}')
    print("\n")
    fold += 1
    i += 1

media_sensivity = 0
media_specificity = 0
media_accuracy = 0

for i in range(10):
    media_sensivity += sens[i]
    media_specificity += spec[i]
    media_accuracy += accu[i]

media_sensivity = media_sensivity/10
media_specificity = media_specificity/10
media_accuracy = media_accuracy/10

print(f'Sensivity media: {media_sensivity}')
print(f'Specificity media: {media_specificity}')
print(f'Accuracy media: {media_accuracy}')

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
plt.title(f'ROC Chart for KNN with k = {k} and {n_fold} folds')
plt.legend(loc="lower right")
plt.show()

