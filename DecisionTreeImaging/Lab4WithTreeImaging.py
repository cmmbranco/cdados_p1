import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import collections
import datetime

def preprocessData(df):
    label_encoder = LabelEncoder()
    dummy_encoder = OneHotEncoder()
    pdf = pd.DataFrame()
    for att in df.columns:
        if df[att].dtype == np.float64 or df[att].dtype == np.int64:
            pdf = pd.concat([pdf, df[att]], axis=1)
        else:
            df[att] = label_encoder.fit_transform(df[att])
            # Fitting One Hot Encoding on train data
            temp = dummy_encoder.fit_transform(df[att].values.reshape(-1,1)).toarray()
            # Changing encoded features into a dataframe with new column names
            temp = pd.DataFrame(temp,
                                columns=[(att + "_" + str(i)) for i in df[att].value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the data frame
            temp = temp.set_index(df.index.values)
            # adding the new One Hot Encoded varibales to the dataframe
            pdf = pd.concat([pdf, temp], axis=1)
    return pdf

breast_cancer = pd.read_csv("./breast_cancer.csv")

X = breast_cancer.iloc[:,:9]
Y = breast_cancer["Class"]
labels = pd.unique(Y)

Y=Y.replace("'recurrence-events'",1)
Y=Y.replace("'no-recurrence-events'",0)


X1 = preprocessData(X)

clf = DecisionTreeClassifier(min_samples_leaf = 15)
rclf=  RandomForestClassifier()

skf = StratifiedKFold(n_splits = 10)

acc =[]
for train_index, test_index in skf.split(X1, Y):
    #print("TRAIN:", train_index, "\nTEST:", test_index)
    X_train = X1.iloc[train_index,:] 
    X_test = X1.iloc[test_index,:]
    y_train = Y.iloc [train_index] 
    y_test = Y.iloc[test_index]

    clf = clf.fit(X_train, y_train)
    rclf = rclf.fit(X_train, y_train)
    acc1= clf.score(X_test,y_test)*100.0
    acc.append(acc1)
    #acc2 = rclf.score(X_test,y_test)

    print('The accuracy of CART was: {}'.format(acc1))
    #print('The accuracy of Random Forest was: {}'.format(acc2))
    #print(clf.decision_path)

std_dev = np.std(acc)

print("The standard deviation is: {}".format(std_dev))



dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    
colors = ('turquoise', 'orange') #convém mudar as cores
edges = collections.defaultdict(list)
    
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
    
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
    


filename = 'bcancer' + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") +'.png'

graph.write_png(filename)
"""
predictions = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test,predictions).ravel()
    
total= tn + fp + fn + tp
    
acc =((tn+tp)/total)*100.0 
    
score = clf.score(X_test,y_test)
"""
