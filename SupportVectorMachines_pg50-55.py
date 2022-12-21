'''A SVM is a classifier that is defined using a separating hyperplane between the classes. This hyperplane is the N-dimensional version of a line. Given labeled training data
and a binary classification problem, the SVM finds the optimal hyperplane that separates the training data into two classes. This can easily be extended to the problem with N classes.

Consider a two-dimensional case with two classes of points. Given that it's 2D, we only have to deal with points and lines in a 2D plane. This is easier to visualize than vectors and 
hyperplanes in a high-dimensional space. This is a simplified version of the SVM problem, but it is important to understand it and visualize it before we can apply it to 
high-dimensional data.'''

'''Consider the figure in the book, in which there are two classes of points and we want to find the optimal hyperplane to separate them.'''
# The solid line indicates the best possible hyperplane, because as a separator, it maximizes the distance of each point from the separating line.
# The points on the dotted lines are called Support Vectors, which are data points closest to the decision boundary, having the greatest influence on the boundary position.
# The perpendicular distance between the two dotted lines is called the maximum margin.


### Classifying income data using SVMs
'''Build a SVM classifier to predict the income bracket of a given person based on 14 attributes. The goal is to see where the income is higher or lower than $50k per year. Hence,
this is a BINARY CLASSIFICATION PROBLEM. We will be using the census income dataset at https://archive.ics.uci.edu/ml/datasets/Census+Income. Each datapoint is a mixture of words
and numbers. We cannot convert everything using label encoder because numerical data is valuable. Hence, we need to use a combination of label encoders and raw numerical data to 
build an effective classifier.'''

# Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
# from sklearn import cross_validation ; cross_validation is no longer available
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split # Have to add this for later

# Input file containing data
input_file = 'Desktop\AI with Python\income_data.txt' # Read in file and store it as 'input_file'.

# Read the data
X = [] # creates an empty list called X. This list will be used to store the input data for the machine learning model.
y = [] # creates an empty list called y. This list will be used to store the target labels for the machine learning model.
count_class1 = 0 # creates an integer variable called count_class1 and assigns it the value 0. This variable will be used to count the number of data points in class 1.
count_class2 = 0 # similar to the previous line, but it creates a variable called count_class2 and assigns it the value 0. This variable will be used to count the number of data points in class 2.
max_datapoints = 25000 # creates an integer variable called max_datapoints and assigns it the value 25000. This variable specifies the maximum number of data points that will be read from the input file.

# Open the file and start reading the lines
with open(input_file, 'r') as f: # opens the file specified by the input_file variable in read mode and assigns the file object to the f variable. The with statement is used to open the file in a context in which it will be automatically closed after the block of code inside the with statement is executed.
    for line in f.readlines(): # begins a loop that iterates over the lines in the file. The readlines method is used to read all the lines in the file and return them as a list.
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints: # checks if the number of data points in class 1 and class 2 is greater than or equal to the maximum number of data points specified by the max_datapoints variable. If this condition is true, the loop will be terminated by the break statement.
            break 

        if '?' in line: # checks if the string '?' is present in the current line of the file. If this condition is true, the loop will skip the rest of the code in the loop body and move on to the next iteration by using the continue statement. This is used to skip lines that contain missing or invalid data.
            continue

        '''Each line is comma separated, so we need to split it accordingly. The last element in each line represents the label. Depending on that label, we will assing it 
        to a class:'''
        data = line[:-1].split(', ') #  take the string stored in line, removing the last character, splitting the resulting string into a list of substrings using a comma and space as the delimiter, and storing the resulting list in the variable data.

        if data[-1] == '<=50K' and count_class1 < max_datapoints: # checking if two conditions are true: (1) the last element of the list stored in the data variable is equal to the string '<=50K', and (2) the value of the variable count_class1 is less than the value of the variable max_datapoints. If both of these conditions are true, then the code block following the if statement will be executed.
            X.append(data) # appending the list stored in the data variable to the end of the list stored in the X variable. This means that data will be added to the end of X as a new element.
            count_class1 += 1 # incrementing the value of count_class1 by 1. This means that the value of count_class1 will be increased by 1 each time this line is executed.

        if data[-1] == '>50K' and count_class2 < max_datapoints: # similar to the first block, but checking if the last element of data is equal to the string '>50K' and if count_class2 is less than max_datapoints.
            X.append(data) # still appending data to the end of X
            count_class2 += 1 # incrementing the value of count_class2 by 1 instead of count_class1.

# Convert to numpy array
X = np.array(X) # Convert list into numpy array.

'''We need to convert all string data into numeric data, while leaving numbers as they are. We will end up with multiple label encoders to keep track of all of them.'''
# Convert string data to numerical data
label_encoder = [] # create an empty list called label_encoder. This list will be used later to store instances of the LabelEncoder class from the sklearn.preprocessing module.
X_encoded = np.empty(X.shape) # uses the NumPy empty() function to create a new NumPy array with the same shape as the array stored in X, but with uninitialized (random) values. The resulting array is stored in the variable X_encoded.
for i, item in enumerate(X[0]): # start a for loop that will iterate over the elements in the first row of the X array. The loop variable i will contain the index of each element, and the variable item will contain the value of each element.
    if item.isdigit(): # check if the current element (item) is a digit, meaning it is a numerical value. If this condition is true, the code block following the if statement will be executed.
        X_encoded[:, i] = X[:, i] # assign the value of the current element in the X array to the corresponding element in the X_encoded array. This is done by selecting the i-th column of both arrays using the : operator.
    else:
        label_encoder.append(preprocessing.LabelEncoder()) # create a new instance of the LabelEncoder class from the sklearn.preprocessing module and appending it to the label_encoder list. The LabelEncoder class is used to encode categorical data, meaning data that consists of strings or labels rather than numerical values.
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i]) # use the fit_transform() method of the LabelEncoder instance stored in the last element of the label_encoder list to encode the current column of the X array. The result of this transformation is then assigned to the corresponding column of the X_encoded array.

X = X_encoded[:, :-1].astype(int) # assign the values in the X_encoded array to the X variable, but only for the columns up to the second-to-last one (:-1). The resulting array is then converted to an integer array using the astype() method.
y = X_encoded[:, -1].astype(int) # similar to the previous line, but it is selecting the last column of the X_encoded array and assigning it to the y variable. The resulting array is then converted to an integer array using the astype() method.

# Creat SVM classifier with a linear kernel:
classifier = OneVsOneClassifier(LinearSVC(random_state=0)) # creates an instance of the OneVsOneClassifier class that uses a LinearSVC classifier to build a set of binary classifiers for each pair of classes in a dataset, with the random_state parameter set to 0, and assigns it to the classifier variable.

# Perform cross-validation using 80/20 split for training/testing, and predict output for training data.
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=5) ; cross_validation no longer supported
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5) # uses the train_test_split() function from the sklearn.model_selection module to split the data stored in the X and y variables into training and test sets, with the test set comprising 20% of the data (test_size=0.2) and the random_state parameter set to 5, and assigns the resulting training and test sets to the X_train, X_test, y_train, and y_test variables.
classifier = OneVsOneClassifier(LinearSVC(random_state=0)) #creates an instance of the OneVsOneClassifier as before, reinitializing it for the new context.
classifier.fit(X_train, y_train) # uses the fit() method of a classifier object to train the model on the training data stored in the X_train and y_train variables.
y_test_pred = classifier.predict(X_test) # uses the predict() method of a classifier object to generate predictions for the test data stored in the X_test variable, and assigns the resulting predictions to the y_test_pred variable.

# Comput F1 score for the classifier
#f1 = cross_validation.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

# Predict output for a test datapoint
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Encode test datapoint
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform(input_data[i]))
        count += 1 

input_data_encoded = np.array(input_data_encoded)

# Run classifier on encoded datapoint and print output
predicted_class = classifier.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicted_class)[0])