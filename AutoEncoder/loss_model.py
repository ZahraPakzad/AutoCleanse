import torch
import torch.nn as nn

def loss_CEMSE(input, outputs, encoder, scaler, continous_columns=None, categorical_columns=None):
    """
     @brief Calculates cross entropy loss and mean square error loss of a dataframe correspondingly to continous and categorical columns
     @param input: The input tensor which is a batch of dataframe rows
     @param outputs: The output which is the autoencoded input
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param encoder: The encoder used to one-hot encode the categorical input values
     @param scaler: The scaler used to scale the continous input values
     @return The combined CE loss and MSE loss
    """  
    output_categorical = None
    output_continous = None
    column_map = None
    slice_list = []
    encoded_columns = encoder.categories_
    if (categorical_columns is not None and continous_columns is not None):
        column_map = {column: encoded_columns[i] for i, column in enumerate(categorical_columns)} # Map categorical columns with onehot subcolumns
        output_categorical = outputs[:,len(continous_columns):]
        output_continous = outputs[:,:len(continous_columns)]
        for i in list(column_map):
            slice_list.append(column_map[i].shape[0])
    elif (continous_columns is None):
        column_map = {column: encoded_columns[i] for i, column in enumerate(categorical_columns)} # Map categorical columns with onehot subcolumns
        output_categorical = outputs
        for i in list(column_map):
            slice_list.append(column_map[i].shape[0])
    elif (categorical_columns is None):
        output_continous = outputs 

    CEloss = 0
    if (categorical_columns is not None):
        # CE loss for each subcolumn group
        Catcols = {}
        CElosses = {}
        val_CElosses = {}
        for i in range(len(categorical_columns)):
            Catcols[f"_{i}"] = None
            CElosses[f"_{i}"] = None
            val_CElosses[f"_{i}"] = None

        start_index_1h = 0                            # Start index relatives to onehot subcolumns range
        start_index = len(continous_columns)          # Start index relatives to entire dataframe
        for i,size in enumerate(slice_list):
            end_index_1h = start_index_1h + size
            end_index = start_index + size
            Catcols[f"_{i}"] = output_categorical[:,start_index_1h:end_index_1h]
            CElosses[f"_{i}"] = nn.CrossEntropyLoss()(Catcols[f"_{i}"],torch.argmax(input[:,start_index:end_index],dim=1)) # Averaged over minibatch
            start_index_1h = end_index_1h
            start_index = end_index

        if (len(categorical_columns)==0):
            pass
        else:
            for value in CElosses.values():
                CEloss += value

    MSEloss = 0
    if (continous_columns is not None):
        if (len(continous_columns)==0):
            pass
        else:
            MSEloss = nn.MSELoss()(output_continous, input[:,:len(continous_columns)])

    return MSEloss + CEloss