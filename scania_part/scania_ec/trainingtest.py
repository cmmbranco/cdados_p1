import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.model_selection._split import train_test_split



#data = pd.read_pickle('../../scania_pickles/train/scania_train_smoted_split_na_normalized.pkl')

#X = data.iloc[:,1:]
#Y = data['class']

#x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, stratify=Y)


train = pd.read_pickle('../../scania_pickles/train/scania_train_subsampled_split_na_normalized.pkl')

test = pd.read_pickle('../../scania_pickles/test/scania_test_split_na_normalized.pkl')

x_train = train.iloc[:,1:]
y_train = train['class']
x_train = np.asarray(x_train)

x_test = test.iloc[:,1:]
x_test = np.asarray(x_test)
y_test = test['class']


print('bla')
labels = pd.unique(y_test)


#print(data)






k = 3




# 
# # #############################################################################
# # Classification and ROC analysis
# 
# Run classifier with cross-validation and plot ROC curves


clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(x_train, y_train)

 
#tprs = []
#aucs = []
#mean_fpr = np.linspace(0, 1, 100)
 
#i = 0
 

     
# Binarize the output
#y_test_bin = label_binarize(y_test, labels)
#n_classes_nb = y_test_bin_nb.shape[1]
print('predicting')
reses = clf.predict(x_test)
confusion = confusion_matrix(y_test, reses, labels)

trueNeg = confusion[0][0]   
truePos = confusion[1][1]  
    
falseNeg = confusion[1][0]  
falsePos = confusion[0][1]  
         
total = trueNeg + truePos + falseNeg + falsePos
acc = ((truePos+trueNeg)/total) * 100.0
specificity = trueNeg / (trueNeg + falsePos)
sensivity = truePos / (truePos + falseNeg)

print(f"Performances for KNN where")
print(confusion)
print(f'number of predictions was {total}')
print(f'accuracy was {acc}')
print(f'specificity rate was {specificity}')
print(f'sensivity rate was {sensivity}')
print("\n")
#acc = accuracy_score(y_test, reses)
#print(acc)


# #probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
# # Compute ROC curve and area the curve
# fpr, tpr, thresholds = roc_curve(y_test_bin, probas_[:, 1])
# tprs.append(interp(mean_fpr, fpr, tpr))
# tprs[-1][0] = 0.0
# roc_auc = auc(fpr, tpr)
# aucs.append(roc_auc)
# plt.plot(fpr, tpr, lw=1, alpha=0.3,
#          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
# 
# i += 1
#      
#      
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#          label='Random', alpha=.8)
#  
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)
#  
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                  label=r'$\pm$ 1 std. dev.')
#  
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'ROC Chart for KNN with k = {k} and {n_fold} folds')
# plt.legend(loc="lower right")
# plt.show()