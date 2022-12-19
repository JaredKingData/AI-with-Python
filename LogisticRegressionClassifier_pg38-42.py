'''Logistic regression is a technique to explain the relationship between input variables (assumed to be independant, or IV) and output variables (referred to as the 'dependent
variable', or DV). The dependent variable can take only a fixed set of values which correspond to the classes of the classification problem. 

The goal is to identify the relationship between the IV and the DVs by estimating the probabilities using a logistic function. This logistic function is a sigmoid curve, used to 
build the function with various parameters, and is closely related to the generalized linear model (GLM) analysis, where we try to fit a line to a bunch of points to minimize error.
We use logistic regression instead of linear regression. Logistic regression is not actually a classification technique by itself, but we commonly use it to facilitate classification
in ML because of its simplicity.'''

'''Here's a brief overview of how logistic regression works:
Logistic regression is a type of linear classifier that is used to predict a binary outcome (e.g., 0 or 1, True or False). It works by finding the line that best separates the 
data points into two classes (e.g., positive and negative). The logistic function is used to transform the output of the linear model into a probability between 0 and 1. 
This probability can then be used to predict the class.For example, suppose we have a dataset with two features (x1 and x2) and a binary outcome (y). We can fit a logistic
regression model to the data by finding the line that best separates the data points into two classes. We can then use the model to predict the class of a new data point by 
computing the probability that it belongs to the positive class. If the probability is above a certain threshold (e.g., 0.5), we classify the data point as positive, otherwise 
we classify it as negative.'''

'''Let's build a classifier using logistic regression with the Tkinter package.'''

# Import relevant packages
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
# *** MAKE SURE YOU HAVE Tkinter INSTALLED ON THIS SYSTEM
from utilities import visualize_classifier

# Define sample input data 
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

'''We will train the classifier usint the labeled datay, but first we create the logistic classifier object.'''
# create the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
'''This code creates a logistic regression classifier object using the LogisticRegression class from the linear_model module in Python's sklearn library. The solver argument 
specifies the algorithm to use for optimization. In this case, it is set to 'liblinear', which is a library for large-scale linear classification. The C argument is a hyperparameter 
that controls the strength of the regularization. A smaller value of C will result in stronger regularization.'''

# Train the classifier with the data that we defined earlier
classifier.fit(X, y)

# Visualize the performance of the classifier
visualize_classifier(classifier, X, y) # This code looks at the boundaries of the classes

# Run it again with a bigger C value
classifier = linear_model.LogisticRegression(solver='liblinear', C=100)
classifier.fit(X, y)
visualize_classifier(classifier, X, y)
'''The boundaries are better in the second figure compared to the first.'''