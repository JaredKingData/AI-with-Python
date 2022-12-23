import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# load data from input file
X = np.loadtxt('data_clustering.txt', delimiter=',')

# estimate the bandwidth of X using quantile (higher = less number of clusters)
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# cluster data with MeanShift (train the Mean Shift clustering model using the estimated bandwidth)
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# extract the centers of clusters
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

# extract the number of clusters
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =")

# plot the points and cluster centers
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    # plot points that belong to the current cluster
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='black')
    
    # plot the center of the current cluster
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
             markerfacecolor='black', markeredgecolor='black',
             markersize=15)
    
plt.title('Clusters')
plt.show()