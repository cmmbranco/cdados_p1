import pandas as pd
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np



nsplits = 3
k_values = (3,5,9,17,33,65,399,401)

###Main LOOP

data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")


"""setup knn for 20 split k-fold with na = 0"""

data = data.replace({'na': '0'}, regex=True)


X=data.iloc[:,1:]
X = np.asarray(X)

Y=data['class']
Yvals = pd.unique(Y)


kf = StratifiedKFold(n_splits=nsplits, random_state=None, shuffle=False)
kf.get_n_splits(data)


for k in k_values:
    
    scores = []
    acc = 0.0
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    print(f"training a new model with K={k}")
    for train_index, test_index in kf.split(X,Y):
        
        print("Training a new split")
        
        #data prep
        for x in train_index:
            x_train.append((X[x]))
            y_train.append(Y[x])
            
        for x in test_index:
            x_test.append((X[x]))
            y_test.append(Y[x])
    
    
    
        clf = KNeighborsClassifier(n_neighbors=k)
        model = clf.fit(x_train, y_train)
        results = model.predict(x_test)
        score = accuracy_score(y_test,results)
        
        scores.append(score)
    print(scores)
        
    median = np.median(scores)
    print(f"median for kneighbors = {k} and n-folds = {nsplits} was {median}")
    
    standdev= np.std(scores)
    print(f"standard deviation for kneighbors = {k} and n-fiks was {standdev}")
