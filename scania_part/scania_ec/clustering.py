from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
import numpy as np
import itertools
from scipy.spatial import distance_matrix

def euclid(vec1, vec2):
    if len(vec1) == len(vec2):
        sum = 0
        for pos in range(len(vec1)):
            dif = vec1[pos]-vec2[pos]
            square_dif = dif*dif
            sum += square_dif
            
        ret = np.sqrt(sum)
        
        return ret
    else:
        print('vector lengths do not match')
        
        
data = pd.read_pickle('../../scania_pickles/train/scania_train_subsampled_split_na_normalized.pkl')

Y = data['class']
X = data.iloc[:,1:]


labels = pd.unique(Y)

count = 0
rep = Y
for label in labels:
    rep = rep.replace(label, count)
    print(rep)
    count += 1
 
print(rep)

kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6)

a = kmeans.fit(X)
y_kmeans = kmeans.predict(X)



plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);



plt.savefig('kmeans.png')




print('Rand score for kmeans')

score = adjusted_rand_score(Y, y_kmeans)

print(score)

d = distance_matrix(X, X)

print('running spectral')
from sklearn.cluster import SpectralClustering
sc = SpectralClustering(2, affinity='precomputed', n_init=100,
                        assign_labels='kmeans')
reses = sc.fit_predict(d)  

print('Rand score for spectral')

score = adjusted_rand_score(Y, reses)


print('silhouette')
a = silhouette_score(d, y_kmeans, metric='euclidean', sample_size=None, random_state=None)


print(a)




