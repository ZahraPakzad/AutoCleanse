from pandas import read_csv, set_option, get_dummies, DataFrame
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from numpy import mean, max, prod, array, hstack
from numpy.random import choice
from matplotlib.pyplot import barh, yticks, ylabel, xlabel, title, show, scatter, cm, figure, imshow
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import display

def softmax(input, encoder, continous_columns, categorical_columns, device):
    """
     @brief Computes softmax activations of a dataframe correspondingly to continous and categorical columns
     @param input: The input tensor
     @param encoder: The onehot encoder used to encode the input
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param device: can be "cpu" or "cuda"
    """
    # Map categorical columns with onehot subcolumns
    encoded_columns = encoder.categories_
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

def argmax(input, encoder, continous_columns, categorical_columns, device):
    """
     @brief Computes argmax activations of a dataframe correspondingly to continous and categorical columns
     @param input: The input tensor
     @param encoder: The onehot encoder used to encode the input
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param device: Can be "cpu" or "cuda"
    """
    # Map categorical columns with onehot subcolumns
    encoded_columns = encoder.categories_
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