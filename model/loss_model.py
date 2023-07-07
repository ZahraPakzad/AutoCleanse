import torch
import torch.nn as nn

def loss_CEMSE(input, outputs, continous_columns, categorical_columns, encoder, scaler):
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
    # Map categorical columns with onehot subcolumns
    encoded_columns = encoder.categories_
    column_map = {column: encoded_columns[i] for i, column in enumerate(categorical_columns)}
    output_continous = outputs[:,:len(continous_columns)]
    output_categorical = outputs[:,len(continous_columns):]

    # Get indices of onehot subcolumns groups
    slice_list = []
    for i in list(column_map):
        slice_list.append(column_map[i].shape[0])

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

    CEloss = 0
    if (len(categorical_columns)==0):
        pass
    else:
        for value in CElosses.values():
            CEloss += value

    MSEloss = 0
    if (len(continous_columns)==0):
        pass
    else:
        MSEloss = nn.MSELoss()(output_continous, input[:,:len(continous_columns)])

    return MSEloss + CEloss