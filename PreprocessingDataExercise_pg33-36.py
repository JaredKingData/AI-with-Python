'''The real world deals with a ton of raw data. Machine Learning (ML) algorithms expect data to be formatted in a certain way before they start the training process. 
to prepare data for ML algorithms to ingest, requires preprocessing to convert it into the right format.'''

# import relevant packages
import numpy as np
from sklearn import preprocessing

# define sample data
input_data = np.array([[5.1, -2.9, 3.3],
                        [-1.2, 7.8, -6.1],
                        [3.9, 0.4, 2.1],
                        [7.3, -9.9, -4.5]])

# let's take a look
print(input_data)     

'''We will be going over several different preprocessing techniques, including binarization, mean removal, scaling, and normalization.'''


# Binarization
'''This process is used to convert numeric values into boolean values. We will use an builtin method to binarize imput data using 2.1 as the threshold value.'''
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data) # Create a new dataset called 'data_binarized' by transforming all values to 1s or 0s based on 2.1 thresh
print("\nBinarized data:\n", data_binarized) # Print the results

'''Result
Binarized data:
 [[1. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [1. 0. 0.]]
This shows that all the values that were previously above 2.1 are now set to 1, whereas all values at or below 2.1 were converted to 0.'''


# Mean Removal
'''One common preprocessing technique used in ML is removing the mean from our feature vector, so that each feature is cetered on zero (centering).
We do this to remove potential biases from the feature in our feature vector.'''

# Print mean and standard deviation
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0)) # Print the mean of the input_data, setting axis=0 means that we will take row means across columns (as opposed to axis=1 which would take column means across rows)
print("Standard Deviation =", input_data.std(axis=0)) # Print the standard deviation of the input_data

'''Result
BEFORE:
Mean = [ 3.775 -1.15  -1.3  ]
Standard Deviation = [3.12039661 6.36651396 4.0620192 ]
This shows what our means and standard deviations for each column look like before we center the data.'''

# Remove mean
data_scaled = preprocessing.scale(input_data) # Scale the data so that it is normalized to reflect the mean removed from each datapoint and then set it equal to data_scaled.
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0)) # Take the mean of the new dataset and print it.
print("Standard Deviation:", data_scaled.std(axis=0)) # Take the new std and print it.

'''AFTER:
Mean = [1.11022302e-16 0.00000000e+00 2.77555756e-17]
Standard Deviation: [1. 1. 1.]
These data is now normalized such that the mean for each column is close to 0, and the standard deviation for each column is set to 1.'''


# Scaling
'''The value of each featur in the feature vector can vary between many random values. It is important to scale those features so as to 'level the playing field' for 
the ML algorithm to train on. We don't want any artificially large or small features due to the nature of measurements.'''

# Min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1)) # Create a scaler such that the minimum value is set to 0 and the maximum value is set to 1.
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data) # Fit the scaler to the data and transform it.
print("\nMin max scaled data:\n", data_scaled_minmax)

'''Result
Min max scaled data:
 [[0.74117647 0.39548023 1.        ]
 [0.         1.         0.        ]
 [0.6        0.5819209  0.87234043]
 [1.         0.         0.17021277]]
Each row is now scaled so that the maximum value is 1 and all other values are relative to this value.'''


# Normalization
'''This process is used to modify the values in the feature vector so that they can be measured on a common scale. In ML, there are many forms of normalization, some of the
most common of which aim to modify the values so that they sum up to 1. 'L1 normalization' refers to "Least Absolute Deviations", and works by making sure that the sum of 
absolute values is 1 in each row. 'L2 normalization' refers to "Least Squares", and works by making sure that the sum of squares is 1.

Generally, L1 is considered more robust than L2, because it is resistant to outliers in the data. Often times, data contains outliers that we cannot do anything about, so we 
want to use techniques that can safely and effectively ignore them during calculations. L2 potentially becomes a better choice if we are solving a problem where outliers are 
important.'''

# Normalize data
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1') # Normalize the data using L1 normalization
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2') # Normalize the data using L2 normalization
print("\nL1 normalized data:\n", data_normalized_l1) # Print the new data normalized using L1
print("\nL2 normalized data:\n", data_normalized_l2) # Print the new data normalized using L2

'''Result
'''