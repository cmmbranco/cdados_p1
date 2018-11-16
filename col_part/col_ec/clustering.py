import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE as smt
from scipy.spatial import distance_matrix

#NOTES
# SILHOUETTE BETWEEN [-1,1]
# RAND SCORE BETWEEN [0,1]

###########################
# DATA IO AND PREPARATION #
###########################

#Alterar aqui o nome do ficheiro para trabalhar com os 3 ficheiros base(ainda falta adaptar para receber smoted/normalized)
data = pd.read_csv('../col_dataset/schiller_norm.csv')

X = data.iloc[:,1:63]
Y = data['consensus']

labels = pd.unique(Y)

count = 0
rep = Y
for label in labels:
    rep = rep.replace(label, count)
    print (rep)
    count += 1

print (rep)

##########
# KMEANS #
##########

kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=6)

a = kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)

plt.savefig('kmeans.png')

d = distance_matrix(X, X)

print ('--K-MEANS--')
print('RAND SCORE')
score = adjusted_rand_score(Y, y_kmeans)
print (score)

print ('SILHOUETTE SCORE')
a = silhouette_score(d, y_kmeans, metric='euclidean', sample_size=None, random_state=None)
print (a)


############
# SPECTRAL #
############

d = distance_matrix(X, X)

spectral = SpectralClustering(5, affinity='precomputed', n_init=100, assign_labels='kmeans')

reses = spectral.fit_predict(d)

print ('--SPECTRAL--')
print('RAND SCORE')
score = adjusted_rand_score(reses, Y)
print(score)

print ('SILHOUETTE SCORE')
a = silhouette_score(d, reses, metric='euclidean', sample_size=None, random_state=None)
print (a)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=reses, s=50, cmap='viridis')

plt.savefig('spectral.png')

plt.show()