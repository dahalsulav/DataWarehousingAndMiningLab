# import required libraries
import numpy as np


# define standard scaler function
def standard_scale(data):
    # convert data to numpy array
    data = np.array(data)
    # calculate mean and standard deviation
    data_mean = np.mean(data)
    data_std = np.std(data)
    # apply standard scaling
    scaled_data = (data - data_mean) / data_std
    # return scaled data
    return scaled_data


# get user input for data to be scaled
data = input("Enter data to be scaled separated by commas: ")

# split the input string into a list of numbers
data = list(map(float, data.split(",")))

# call standard scaler function on input data
scaled_data = standard_scale(data)

# print the scaled data
print("Scaled data:", scaled_data)
