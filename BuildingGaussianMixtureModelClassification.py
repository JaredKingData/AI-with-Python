import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from sklearn import datasets
#from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture as GMM
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold

# load the iris dataset
iris = datasets.load_iris()

# split dataset into training and testing (80/20 split)
#indices = StratifiedKFold(iris.target, n_folds=5) ...n_folds was removed, use n_splits instead
#indices = StratifiedKFold(iris.target, n_splits=5)
indices = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

'''Extract the training data:'''
# take the first fold
#train_index, test_index = next(iter(indices))
# iterate over the splits
for train_index, test_index in indices.split(iris.data, iris.target):
    # extract training data and labels
    X_train, y_train = iris.data[train_index], iris.target[train_index]
    
    # extract testing data and labels
    X_test, y_test = iris.data[test_index], iris.target[test_index]

# extracet training data and labels
X_train = iris.data[train_index]
y_train = iris.target[train_index]

# extract testing data and labels
X_test = iris.data[test_index]
y_test = iris.target[test_index]

# extract the number of classes in the training data
num_classes = len(np.unique(y_train))

# Build GMM
#classifier = GMM(n_components=num_classes, covariance_type='full', 
#                 init_params='wc', n_iter=20) ...no longer accepts n_iter argument, and 'init_params' is automatically set to 'wc' by default, so no longer necessary.
classifier = GMM(n_components=num_classes, covariance_type='full')

# initialize the means of the GMM 
classifier.means_ = np.array([X_train == i].mean(axis=0) 
                             for i in range(num_classes))

# train the GMM classifier
classifier.fit(X_train)

# extract eigenvalues and eigenvectors to estimate elliptical boundaries
#plt.figure() 
#colors = 'bgr'
#for i, color in enumerate(colors):
    # extract eigenvalues and eigenvectors
#    eigenvalues, eigenvectors = np.linalg.eigh(
#        classifier._get_covars()[i][:2, :2]) ..._get_covars no longer supported, use 'covariances_' instead
    
# extract eigenvalues and eigenvectors to estimate elliptical boundaries
plt.figure()
colors = 'bgr'
for i, color in enumerate(colors):
    # extract the covariance matrix for the i-th component
    covar = classifier.covariances_[i, :2, :2]
    
    # extract eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covar)
    
    # normalize the first eigenvector
    norm_vec = eigenvectors[0] / np.linalg.norm(eigenvectors[0])
    
    # extract the angle of tilt (need to be rotated to accurately show distribution)
    angle = np.arctan2(norm_vec[1], norm_vec[0])
    angle = 180 * angle / np.pi
    
    # scaling factor to magnify ellipses (eigenvalues control the size)
    # (random value chosen to suit our needs)
    scaling_factor = 8
    eigenvalues *= scaling_factor
    
    # draw the ellipses
    ellipse = patches.Ellipse(classifier.means_[i, :2], 
                              eigenvalues[0], eigenvalues[1], 180 + angle,
                              color=color)
    axis_handle = plt.subplot(1, 1, 1)
    ellipse.set_clip_box(axis_handle.bbox)
    ellipse.set_alpha(0.6)
    axis_handle.add_artist(ellipse)
    
# plot the data
colors = 'bgr'
for i, color in enumerate(colors):
    cur_data = iris.data[iris.target == i]
    plt.scatter(cur_data[:,0], cur_data[:,1], marker='o',
                facecolors='none', edgecolors='black', s=40,
                label=iris.target_names[i])
    
    # overlay test data on figure
    test_data = X_test[y_test == i]
    plt.scatter(test_data[:,0], test_data[:,1], marker='s',
                facecolors='black', edgecolors='black', s=40,
                label=iris.target_names[i])
    
# compute predictions for training and testing data
y_train_pred = classifier.predict(X_train)
accuracy_training = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
print('Accuracy on training data =', accuracy_training)

y_test_pred = classifier.predict(X_test)
accuracy_testing = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print('Accuracy on testing data =', accuracy_testing)

plt.title('GMM classifier')
plt.xticks(())
plt.yticks(())

plt.show()

 
    
    
    
    
    
    
    
    
