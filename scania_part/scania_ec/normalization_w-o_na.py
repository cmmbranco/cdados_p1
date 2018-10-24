import pandas as pd
from sklearn.preprocessing import normalize

data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv", na_values='na')

X=data.iloc[:,1:]

X=X.fillna(-1)

X_normalized = normalize(X,norm='max')
