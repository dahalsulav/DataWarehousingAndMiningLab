# import required libraries
import numpy as np


# define min-max scaler function
def minmax_scale(data):
    # convert data to numpy array
    data = np.array(data)
    # calculate min and max values
    data_min = np.min(data)

    data_max = np.max(data)
    if data_min == data_max:
        # return zero
        return np.zeros(data.shape)
    # apply min-max scaling
    scaled_data = (data - data_min) / (data_max - data_min)
    # return scaled data
    return scaled_data


# get user input for data to be scaled
data = input("Enter data to be scaled separated by commas: ")

# split the input string into a list of numbers
data = list(map(float, data.split(",")))

# call min-max scaler function on input data
scaled_data = minmax_scale(data)

# print the scaled data
print("Scaled data:", scaled_data)
