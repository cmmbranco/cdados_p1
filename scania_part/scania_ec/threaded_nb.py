import logging
from threading import Thread

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

results = {}

def threadedNB(k,n_fold,output):

    try:
        ###Main LOOP

        data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")
        data = data.replace({'na': '0'}, regex=True)
        
        
        data2 = pd.read_csv("../scania_dataset/aps_failure_test_set.csv")
        data2 = data2.replace({'na': '0'}, regex=True)
        
        X = data.iloc[:,1:]
        X = np.asarray(X)
        
        X1 = data2.iloc[:,1:]
        X1 = np.asarray(X1)
        
        Y = data['class']
        Yvals = pd.unique(Y)
        
        Y1 = data2['class']
            
        scores = []
        x_train = X
        y_train = Y
        x_test = X1
        y_test = Y1
        
#         for train_index, test_index in kf.split(X,Y):
#             
#             #print("Training a new split")
#                 
#             #data prep
#             for x in train_index:
#                 x_train.append((X[x]))
#                 y_train.append(Y[x])
#                 
#             for x in test_index:
#                 x_test.append((X[x]))
#                 y_test.append(Y[x])
            
            
        clf = GaussianNB()

        model = clf.fit(x_train, y_train)

        results = clf.predict(x_test)
            
        score = accuracy_score(y_test,results)
        print("model score acquired")    
        scores.append(score)
                
        median = np.median(scores)
        #print(f"median for kneighbors = {k} and n-folds = {nsplits} was {median}")
            
        standdev= np.std(scores)
        #print(f"standard deviation for kneighbors = {k} and n-fiks was {standdev}")
        
        output[k] = (n_fold,median,standdev)
        
    except:
        logging.error('threaded NB with k = {k} failed')
        output[k] = {}
        
    return True





#create a list of threads
threads = []


#,33,65,399,401)
n_folds = 3
result={}

process = Thread(target=threadedNB, args=[1,n_folds, result])
print(f"Starting NB: 1")
process.start()
threads.append(process)

    
for process in threads:
    process.join()
        
    
print(result)


