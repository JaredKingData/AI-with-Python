'''Performing classification usually deals with many labels that can be in the form of words, numbers, or something else. ML functions in sklearn expect them to be numbers, so if 
they are numbers, then they can already be used to start training. However, that is not always the case.

In the real world, labels take the form of words because words are readable to humans. To convert word labels into numbers, we use a label encoder to transform word labels into 
numerical forms, so that the algorithms can operate on our data.'''

# import relevant stuff
import numpy as np
from sklearn import preprocessing

# define sample input labels
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# create label encoder and fit the labels
encoder = preprocessing.LabelEncoder() # create the label encoder
encoder.fit(input_labels) # use it to fit the labels

# print the mapping between words and numbers
print("\nLabel Mapping:")
for i, item in enumerate(encoder.classes_): # for i (index number), item (current element)
    print(item, '-->', i) # print the current element --> index number for each instance

'''Result
Label Mapping:
black --> 0
green --> 1
red --> 2
white --> 3
yellow --> 4
It looks like it assigned each label a numeric value.'''

# Encode a set of randomly ordered labels using the encoder
test_labels = ['green', 'red', 'black'] # store test labels 
encoded_values = encoder.transform(test_labels) # store a new list by converting each label using the numerical assignment from before
print("\nLabels =", test_labels) # print each label
print("Encoded values =", list(encoded_values)) # print each label's numeric value

'''Result
Labels = ['green', 'red', 'black']
Encoded values = [1, 2, 0]
We were able to correctly label new data with the numeric values that were assigned previously.'''

# Decode a random set of numbers using the encoder
encoded_values = [3, 0, 4, 1] # store test numbers
decoded_list = encoder.inverse_transform(encoded_values) # store a new list using the inverse transform to create a label for each number
print("\nEncoded Values =", encoded_values) # print the encoded values as numbers
print("Decoded labels =", list(decoded_list)) # print the decoded list as labels

'''Results
Encoded Values = [3, 0, 4, 1]
Decoded labels = ['white', 'black', 'yellow', 'green']
We were able to successfully extract our labels from given numeric data.'''