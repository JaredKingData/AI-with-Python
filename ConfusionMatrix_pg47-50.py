'''A Confusion matrix is a figure or table used to describe performance of a classifier. It is usually extracted from a test dataset for which the ground truth is known. 
We compare each class with every other class and see how many samples are misclassified. During construction of this table, we actually come across several key metrics that 
very important in the field of ML. Let's consider a binary classification case where the output is either 0 or 1:'''

# True positives: Samples where we predicted 1 as the output and the ground truth is also 1.
# True negatives: Samples where we predicted 0 as the output and the ground truth is also 0.
# False positives: Samples where we predicted 1 as the output, but the ground truth is actually 0.
# False negatives: Samples where we predicted 0 as the output, but the ground truth is actually 1.

'''Depending on the problem, we may optimize our algorithm to reduce the false positive or false negative rates. For example, in biometric identification system, it is very important
to avoid false positives, because the wrong people might get access to sensitive information. Let's see how to create a confusion matrix:'''

# import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define sample labels
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Create confusion matrix
confusion_mat = confusion_matrix(true_labels, pred_labels)

# Visualize confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray) # Uses the imshow function from matplotlib to display an image representing the confusion matrix stored in the confusion_mat variable. The interpolation argument controls the way the values in the matrix are interpolated to display the image. The cmap argument specifies the colormap to use for the image. In this case, the gray colormap is used, which will display the image in grayscale.
plt.title('Confusion matrix') # Add a title to the plot using the title function, which will be displayed above the plot.
plt.colorbar() # Add a colorbar to the plot using the colorbar function. A colorbar is a scale that shows the mapping of colors to values. It is useful for understanding how the values in the confusion matrix map to colors in the plot.
ticks = np.arange(5) # Create an array of integers from 0 to 4 using the arange function from NumPy. This array will be used as the tick marks for the x-axis and y-axis.
plt.xticks(ticks, ticks) # Add tick marks to the x-axis using the xticks function. The first argument specifies the position of the tick marks, and the second argument specifies the labels for the tick marks. In this case, the tick marks and labels are the same, so the same array is used for both arguments.
plt.yticks(ticks, ticks) # Similar to the previous line, but it adds tick marks and labels to the y-axis instead of the x-axis.
plt.ylabel('True labels') # Add a label to the y-axis using the ylabel function. The label will be displayed alongside the y-axis.
plt.xlabel('Predicted labels') # Similar to the previous line, but it adds a label to the x-axis.
plt.show() # Displays the plot using the show function.

# Print the Classification Report:
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4'] # Creates a list of strings that represent the class labels for the classification task. The list has five elements, corresponding to five possible classes.
print('\n', classification_report(true_labels, pred_labels, target_names=targets)) # Use the classification_report function from sklearn.metrics to generate a summary of the performance of a classification model. The true_labels and pred_labels variables contain the true class labels and predicted class labels, respectively. The target_names argument specifies the names of the classes, which are provided in the targets list. The \n string is added at the beginning of the line to add a newline character before the classification report is printed. This helps to visually separate the report from any other output that may have been printed previously.

'''White squares indicate higher values, whereas black indicates lower values as seen on the color map slider. In an ideal scenario, the diagonal squares will be all white and 
everything else will be black, indicating 100% accuracy.'''

'''Result: 
               precision    recall  f1-score   support
     Class-0       1.00      0.67      0.80         3
     Class-1       0.33      1.00      0.50         1
     Class-2       1.00      1.00      1.00         2
     Class-3       0.67      0.67      0.67         3
     Class-4       1.00      0.50      0.67         2

    accuracy                           0.73        11
   macro avg       0.80      0.77      0.73        11
weighted avg       0.85      0.73      0.75        11

'''
