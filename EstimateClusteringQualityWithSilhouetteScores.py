# import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

# load data from input file
X = np.loadtxt('data_quality.txt', delimiter=',')

# initialize variables
scores = []
values = np.arange(2, 10)

# iterate through the defined range
for num_clusters in values:
    # train the KMeans clustering model
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
    
    # estimating the silhouette score for the current clustering model using euclidean distance metric
    score = metrics.silhouette_score(X, kmeans.labels_, 
                                     metric='euclidean', sample_size=len(X))

    # print silhouette score for current value
    print("\nNumber of clusters =", num_clusters)
    print("Silhouette score =", score)
    scores.append(score)

# plot silhouette scores
plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('Silhouette score vs number of clusters')

# extract best score and optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters =', num_clusters)

# plot these data
plt.figure()
plt.scatter(X[:,0], X[:,1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
