import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO  
#from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import collections
import datetime

data = pd.read_pickle('../../scania_pickles/train/scania_train_smoted_split_na_normalized.pkl')
        
X=data.iloc[:,1:]
X = np.asarray(X)
        
Y=data['class']
labels = pd.unique(Y)
Y = np.asarray(Y)

clf = DecisionTreeClassifier()
#rclf=  RandomForestClassifier()

clf.fit(X,Y)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  



#printing tree

colors = ('turquoise', 'orange') #convém mudar as cores
edges = collections.defaultdict(list)
    
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
    
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
    
graph.write_png('cart_smoted_visualization.png')



