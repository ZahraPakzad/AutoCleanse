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
import random
from ordered_set import OrderedSet
from sklearn.linear_model import LinearRegression
from dataloader import PlainDataset, DataLoader

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
    
    df = dataframe.copy()
    # Calculate the number of elements to replace with NaN
    num_elements_to_replace = int(df.size * ratio)

    # Flatten the DataFrame and select random positions to replace with NaN
    flat_data = df.to_numpy().flatten()
    indices_to_replace = np.random.choice(flat_data.size, num_elements_to_replace, replace=False)
    flat_data[indices_to_replace] = np.nan

    # Reshape the flat data back to the original shape
    df[:] = flat_data.reshape(df.shape)
    return df 

def generate_random_spike(a, b):
    # Generate a random value in the range [a, b]
    random_value = np.random.uniform(a, b)

    # Determine whether to negate the value
    if np.random.choice([True, False]):
        random_value *= -1

    return random_value


def replace_nan_values(input_df, col= None, continous_columns=[], categorical_columns=[], og_columns=None, method="mean", MAE_pack=None, encoder=None, scaler=None):
    
    if MAE_pack is not None: 
        dirty_dataset = PlainDataset(input_df)
        dirty_loader = DataLoader(dirty_dataset, batch_size=MAE_pack[1], shuffle=False, drop_last=True, collate_fn=MAE_pack[3])

        cleaned_data = MAE_pack[0].clean(
            dirty_loader=dirty_loader,
            test_loader=MAE_pack[4],
            df=input_df,
            batch_size=MAE_pack[1],
            continous_columns=continous_columns, 
            categorical_columns=categorical_columns, # None, since we don't deal with the NAN values of categorical cols here
            og_columns=og_columns,
            onehotencoder=encoder, 
            scaler=scaler,
            device=MAE_pack[2],
            is_mae = True
        )

        input_df = cleaned_data
        return input_df
    
    else: 
        nan_indices = input_df[col].index[input_df[col].isna()]
        df_nonNA = input_df[input_df[col].notna()]
        nonNAN_values = list(df_nonNA[col]) # values of nonNAN
        
        print("other methods")
        if method == "distribution_based":
            # binning the values
            NUM_BINS = 10000  
            bins = pd.cut(nonNAN_values, NUM_BINS, labels = list(range(0, NUM_BINS)))

        
            bins_frequency = {}
            for item in bins:
                for i in range(0, NUM_BINS):
                    if item == i:
                        if i in bins_frequency:
                            bins_frequency[i] += 1
                        else:
                            bins_frequency[i] = 1
                    break

            # random selection of bins
            population = OrderedSet(bins)
            weights = [v/len(bins) for k, v in bins_frequency.items()]
            k = len(nan_indices)

            selected_bins = random.choices(population, weights, k=k)

            # make a dictionary where the bin numbers are the dictionary keys and the values
            # of each bin is listed as the dictionary's value.
            dict_bins = {}

            for item in zip(bins, nonNAN_values):
                if item[0] in dict_bins:
                    dict_bins[item[0]].append(int(item[1]))
                else:
                    dict_bins[item[0]] = [item[1]]

            for i in range(0, len(selected_bins)):
                random_value = random.choices(dict_bins[selected_bins[i]])
                input_df.loc[nan_indices[i], col] = random_value[0]

        elif method =="median": 
            input_df.col.fillna(input_df.col.median(), inplace=True)
        else: # mean
            input_df.col.fillna(input_df.col.mean(), inplace=True)

        return input_df


def ml_model_impute_missing_values(target_column, other_columns):
    
    pass
    

def string2list(input: str):
    output = list(map(int, input.split(',')))
    return output

def string2tupple(input: str):
    output = tuple(map(int, input.split(',')))
    return output
