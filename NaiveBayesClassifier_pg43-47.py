'''Naive Bayes uses Bayes theorem, which describes the probability of an event occurring based on different conditions that are related to that event, to build classifiers. 
We build this classifier by assigning labels to problem instances which are represented as feature vectors. The assumption is that the value of a given feature is independent 
of the value of any other feature (the "naive" part, called independence assumption).'''

'''Given the class variable, we can see how a given feature affects it, regardless of its effect on other features. For example, an animal may be considered a cheetah if it is 
spotted, has 4 legs, has a tail, and runs 70mph. A Naive Bayes classifier considers that each of these features contributes independently to the outcome. The outcome refers to 
the probability that this animal is a cheetah. We don't care about the correlations that may exist between skin patterns, number of legs, presence of tail, and movement speed.'''

# import relavent packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import model_selection

from utilities import visualize_classifier

'''We will be using data_multivar_nb.txt as a source of data, which uses comma separated values in each line.'''
# input file containing data
input_file = 'Desktop\ML with Python/data_multivar_nb.txt' # Have to remember to put the path to the file in, not just the name of the file.

# load the data from this file
data = np.loadtxt(input_file, delimiter = ',')
X, y = data[:, :-1], data[:, -1]

# Create the Naive Bayes Classifier (We will be using the Gaussian Naive Bayes classifier here)
classifier = GaussianNB()

# Train the classifier using the training data
classifier.fit(X, y)

# Predict the values for training data
y_pred = classifier.predict(X)

'''Compute the accuracy of the classifier by comparing the predicted values to the true labels, and then visualize performance'''
# Compute accuracy
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes Classifier =", round(accuracy, 2), "%")
'''This code calculates the accuracy of a classifier by comparing the predicted labels (y.pred) to the true labels (y). The accuracy is calculated as the percentage of instances 
where the predicted label is equal to the true label.

Here's a breakdown of what each line does:
y == y.pred: This compares the predicted labels to the true labels element-wise and returns a boolean array indicating which predictions are correct (True) and which are 
incorrect (False).
(y == y.pred).sum(): This sums up the boolean array, which gives us the total number of correct predictions.
X.shape[0]: This gets the number of rows in the feature matrix X.
100.0 * (y == y.pred).sum() / X.shape[0]: This calculates the accuracy as the percentage of correct predictions.
Finally, the code prints out the accuracy, rounding it to 2 decimal places using the round function.'''

# Visualize performance of the classifier
visualize_classifier(classifier, X, y)

'''Not very robust... we need to perform cross-validation so that we don't use the same training data when we are testing it. Split the data into training and testing subsets. As 
specified by test_size parameter below, we will allocate 80% for training and the remaining 20% for testing. Then we will use the Naive Bayes classifier on the data.'''

# split the data into training and testing data
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=3) # cross_validation module was removed from 0.22 Python's 'sklearn' library
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=3)
'''This code uses the train_test_split function from the cross_validation module in Python's sklearn library to split a dataset into training and test sets. The function takes 
four arguments:

    X: A feature matrix containing the independent variables (also known as predictors or features) of the dataset.
    y: A vector containing the dependent variable (also known as the target or label) of the dataset.
    test_size: The proportion of the dataset to include in the test set. This can be a float between 0.0 and 1.0, or an integer representing the number of test samples.
    random_state: A seed for the random number generator. This can be an integer or None (in which case the random number generator will be initialized with a randomly chosen seed).

The train_test_split function returns four objects:

    X_train: A feature matrix containing the training data.
    X_test: A feature matrix containing the test data.
    y_train: A vector containing the training labels.
    y_test: A vector containing the test labels.

The train_test_split function randomly shuffles the data before splitting it into the training and test sets. This is useful because it ensures that the training and test sets are 
representative of the overall distribution of the data, and it helps prevent overfitting (which is when a model performs well on the training data but poorly on new, unseen data).'''

classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

# Compute the accuracy of the classifier 
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy, 2), "%")

# Visualize the performance of the classifier
visualize_classifier(classifier_new, X_test, y_test)

'''Let's use builtin functions to calculate accuracy, precision, and recall values based on threefold cross validation.'''
num_folds = 3
# accuracy_values = cross_validation.cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
# print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) +"&")

# precision_values = cross_validation.cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
# print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

# recall_values = cross_validation.cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
# print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

# f1_values = cross_validation.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
# print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")

'''Same problem as before with cross_validation. Code is rewritten to be correct.'''
accuracy_values = model_selection.cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = model_selection.cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = model_selection.cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = model_selection.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")