import logging
from threading import Thread
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics.classification import accuracy_score
#from sklearn.model_selection import StratifiedKFold
import numpy as np

results = {}

def threadedRForest(k,n_fold,output):

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
        
        
        #kf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=False)
        #kf.get_n_splits(data)
        
        
            
        scores = []
        x_train = X
        y_train = Y
        x_test = X1
        y_test = Y1

        clf = RandomForestClassifier(n_estimators=k)

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
        
    except Exception as e: 
        print(e)
        
    return True





#create a list of threads
threads = []


#,33,65,399,401)
n_folds = 3
n_estim = (5,10,20,30,50,100)
result={}
for k in n_estim:
    
    
    
    process = Thread(target=threadedRForest, args=[k,n_folds, result])
    print(f"Starting CART with {k} forests")
    process.start()
    threads.append(process)
    
        


for process in threads:
        process.join()
            
        
print(result)


