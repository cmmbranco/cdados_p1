import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from sklearn.neighbors.classification import KNeighborsClassifier

n_fold = 10
k = 57

##########################
# Data IO and generation #
##########################

# Import some data to play with

green_data = pd.read_csv('../col_dataset/green.csv')
hinselmann_data = pd.read_csv('../col_dataset/hinselmann.csv')
schiller_data = pd.read_csv('../col_dataset/schiller.csv')

X_green = green_data.iloc[:,:68]
X_hinselmann = hinselmann_data.iloc[:,:68]
X_schiller = schiller_data.iloc[:,:68]

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

kf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=False)
kf.get_n_splits(green_data)

scores = []
x_train = []
y_train = []
x_test = []
y_test = []

#########################3#########
# Classification and ROC Analysis #
###################################
cv = StratifiedKFold(n_splits=n_fold)
clf = KNeighborsClassifier(n_neighbors=k)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0

for train_index, test_index in kf.split(X_green, Y_green):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = X_green[train_index], X_green[test_index]
    y_train, y_test = Y_green[train_index], Y_green[test_index]

    # Binarize the output
    y_test_bin = label_binarize(y_test, green_labels)
    # n_classes_nb = y_test_bin_nb.shape[1]


    probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test_bin, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

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
plt.title(f'ROC Chart for KNN with k = {k} and {n_fold} folds')
plt.legend(loc="lower right")
plt.show()

