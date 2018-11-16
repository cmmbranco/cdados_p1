import matplotlib.pyplot as plt

######################
# Parameters to pass K #
######################

#rand = [0.2022, 0.5309, 0.2591, 0.2053]
#silhouette = [0.5538, 0.3874, 0.0265, -0.2212]
#kmeans = [2,5,10,20]

#plt.plot(kmeans, rand, 'g', label=r'Rand Score')
#plt.plot(kmeans, silhouette, 'r', label=r'Silhouette Score')
#plt.plot(knn, acc, 'b', label=r'Accuracy')

#plt.xlabel('Nr_Clusters')
#plt.ylabel('Performance (%)')
#plt.title('Performance Chart for Clustering')
#plt.legend(loc="lower right")
#plt.show()

sensivity = [40.266, 45.333, 50.666, 55.466, 61.866, 69.333, 85.600]
specificity = [97.260, 97.465, 97.542, 94.822, 90.297, 91.001, 83.110]
acc = [95.925, 96.243, 96.443, 93.899, 89.631, 90.493, 83.168]
knn = [2, 20, 200, 600, 1000, 2000, 5000]

plt.plot(knn, sensivity, 'g', label=r'Sensivity Rate')
plt.plot(knn, specificity, 'r', label=r'Specificity Rate')
plt.plot(knn, acc, 'b', label=r'Accuracy')

plt.xlabel('Min_Samples_Split')
plt.ylabel('Performance (%)')
plt.title('Performance Chart for CART')
plt.legend(loc="lower right")
plt.show()