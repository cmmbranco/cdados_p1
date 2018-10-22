from sklearn.tree import DecisionTreeClassifier
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize



n_fold = 10

# #############################################################################
# Data IO and generation

# Import some data to play with

data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")
data = data.replace({'na': '-1'}, regex=True)
        
X=data.iloc[:,1:]
X = np.asarray(X)
        
Y=data['class']
labels = pd.unique(Y)
Y = np.asarray(Y)
        
kf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=False)
kf.get_n_splits(data)
        
        
            
scores = []
x_train = []
y_train = []
x_test = []
y_test = []




# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=n_fold)
clf = DecisionTreeClassifier()


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0

for train_index, test_index in kf.split(X,Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, labels)
    #n_classes_nb = y_test_bin_nb.shape[1]

    probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
    #probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
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
plt.title(f'ROC Chart for CART and {n_fold} folds')
plt.legend(loc="lower right")
plt.show()


