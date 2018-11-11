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

data = pd.read_csv('../col_dataset/green.csv')

Y = data['consensus']
X = data.iloc[:,:62]

labels = pd.unique(Y)

count = 0
rep = Y
for label in labels:
    rep = rep.replace(label, count)
    print (rep)
    count += 1

print (rep)

##################
# PRE-PROCESSING #
##################

# Normalization (comment it if want to check results with no normalization)
#X = normalize(X, axis=0, norm='max')

# Resampling (comment it if want to check results with no resampling)
#X_hinselmann, Y_hinselmann = resample(X_hinselmann, Y_hinselmann)

# Smote (comment it if want to check results with no smote)
#smote = smt(ratio='minority')
#X, Y = smote.fit_sample(X, Y)

##########
# KMEANS #
##########

kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6)

a = kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)

plt.savefig('kmeans.png')

d = distance_matrix(X, X)

print('Rand Score for kmeans')
score = adjusted_rand_score(Y, y_kmeans)
print (score)

print ('Silhouette score for kmeans')
a = silhouette_score(d, y_kmeans, metric='euclidean', sample_size=None, random_state=None)
print (a)


print ('---------------------------')
############
# SPECTRAL #
############

d = distance_matrix(X, X)

spectral = SpectralClustering(2, affinity='precomputed', n_init=100, assign_labels='kmeans')

reses = spectral.fit_predict(d)

print('Rand Score for Spectral')
score = adjusted_rand_score(reses, Y)
print(score)

print('Silhouette score for spectral')
a = silhouette_score(d, reses, metric='euclidean', sample_size=None, random_state=None)
print (a)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=reses, s=50, cmap='viridis')

plt.savefig('spectral.png')

#plt.show()