import pandas as pd
from sklearn.preprocessing import normalize

green_data = pd.read_csv('../col_dataset/green.csv')

X = green_data.iloc[:,:68]

X_normalized = normalize(X, norm='max')

print (X_normalized)