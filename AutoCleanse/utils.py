from pandas import read_csv, set_option, get_dummies, DataFrame
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
import numpy as np
from numpy import mean, max, prod, array, hstack
from numpy.random import choice
import torch
import torch.nn.functional as F
import torch.optim as optim

def softmax(input, onehotencoder, continous_columns, categorical_columns, device):
    """
     @brief Computes softmax activations of a dataframe correspondingly to continous and categorical columns
     @param input: The input tensor
     @param onehotencoder: The onehot encoder used to encode the categorical input
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param device: can be "cpu" or "cuda"
    """
    # Map categorical columns with onehot subcolumns
    encoded_columns = onehotencoder.categories_
    column_map = {column: encoded_columns[i] for i, column in enumerate(categorical_columns)}

    # Get indices of onehot subcolumns groups
    slice_list = []
    for i in list(column_map):
        slice_list.append(column_map[i].shape[0])

    start_index = 0
    softmax_column = {}
    output = torch.empty(0).to(device)
    for i,size in enumerate(slice_list):
        end_index = start_index + size
        softmax_column[f"_{i}"] = F.softmax(input[:,start_index:end_index],dim=1)
        output = torch.cat((output,softmax_column[f"_{i}"]),dim=1)
        start_index = end_index

    return output

def argmax(input, onehotencoder, continous_columns, categorical_columns, device):
    """
     @brief Computes argmax activations of a dataframe correspondingly to continous and categorical columns
     @param input: The input tensor
     @param encoder: The onehot encoder used to encode the categorical input
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param device: Can be "cpu" or "cuda"
    """
    # Map categorical columns with onehot subcolumns
    encoded_columns = onehotencoder.categories_
    column_map = {column: encoded_columns[i] for i, column in enumerate(categorical_columns)}

    # Get indices of onehot subcolumns groups
    slice_list = []
    for i in list(column_map):
        slice_list.append(column_map[i].shape[0])

    start_index = 0
    output = torch.empty(0).to(device)
    for i,size in enumerate(slice_list):
        end_index = start_index + size
        argmax_indices = torch.argmax(input[:,start_index:end_index],dim=1)
        argmax_output = F.one_hot(argmax_indices,input[:,start_index:end_index].size(1))
        output = torch.cat((output,argmax_output),dim=1)
        start_index = end_index

    return output

def generate_suffix(layer_sizes,prefix,load_method=None):
    # Convert the list of layer sizes to a list of strings
    layer_sizes_str = [str(size) for size in layer_sizes[1:]]
    
    if (load_method=="bucketfs"):
        autoencoder_name = f'/autoencoder/{prefix}_' + '_'.join(layer_sizes_str) + '.pth'
    elif (load_method=="local"):
        autoencoder_name = f'{prefix}_' + '_'.join(layer_sizes_str) + '.pth'
    elif (load_method==None):
        autoencoder_name = None
    return autoencoder_name

def replace_with_nan(dataframe, ratio, seed):
    if not 0 <= ratio <= 1:
        raise ValueError("Ratio must be between 0 and 1.")
    np.random.seed(seed)
    
    # Calculate the number of elements to replace with NaN
    num_elements_to_replace = int(dataframe.size * ratio)

    # Flatten the DataFrame and select random positions to replace with NaN
    flat_data = dataframe.to_numpy().flatten()
    indices_to_replace = np.random.choice(flat_data.size, num_elements_to_replace, replace=False)
    flat_data[indices_to_replace] = np.nan

    # Reshape the flat data back to the original shape
    dataframe[:] = flat_data.reshape(dataframe.shape)
    return dataframe

def string2list(input: str):
    output = list(map(int, input.split(',')))
    return output

def string2tupple(input: str):
    output = tuple(map(int, input.split(',')))
    return output
