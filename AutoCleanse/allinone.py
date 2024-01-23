import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import joblib
import io

from tqdm import tqdm
from pandas import read_csv, set_option, get_dummies, DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from numpy import mean, max, prod, array, hstack
from numpy.random import choice
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from exasol.bucketfs import Service
from exasol.bucketfs import (
    Service,
    as_bytes,
    as_file,
    as_string,
)
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx].values
        tensor_data = torch.stack([torch.tensor(item, dtype=torch.float32) for item in data])
        return tensor_data, idx


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

def dataSplitter(input_df,train_ratio,val_ratio,test_ratio,random_seed):
    """
     @brief Split dataset into training, validation and test set
     @param train_ratio: float or int ratio of training data
     @param val_ratio: float or int ratio of validation data
     @param test_ratio: float or int ratio of test data
     @param random_seed: for debugging
    """
    # Calculate the sizes of train, validation, and test sets
    assert(train_ratio+val_ratio+test_ratio) == 1,f"Ratio is not equal to 1, got{train_ratio+val_ratio+test_ratio} instead."
    total_size = len(input_df)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split training, validation, and test sets
    X_train, X_temp = train_test_split(input_df, test_size=(val_size + test_size), random_state=random_seed)    
    X_val, X_test = train_test_split(X_temp, test_size=test_size, random_state=random_seed)
    return X_train,X_val,X_test


def dataPreprocessor(input_df,is_train,continous_columns,categorical_columns,scaler,onehotencoder,save=None,file_path=None):
    """
     @brief apply scaling and encoding to the input dataframe
     @param input_df: Input dataframe    
     @param is_train: if the input data is a train set, use fit_transform(), else use transform()
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param scaler: Scaler object
     @param onehotencoder: Onehot encoder object
     @param save: Enable saving fit-transformed scaler and encoder. Can be "BucketFS" or "local"
     @param file_path: Path of the save destination of scaler/encoder
    """
    if (save is not None):
      assert file_path is not None, "file_path must be declared if save is specified."
    if (file_path is not None):
      assert save is not None, "save must be declared if file_path is specified."

    client = BucketFS_client()
    # Preprocess continous columns
    if (is_train == True):
      input_df_scaled = scaler.fit_transform(input_df[continous_columns])
      if (save is not None):          
        if (save=="BucketFS"):
          # Save to BucketFS 
          buffer = io.BytesIO()
          joblib.dump(scaler,buffer)
          client.upload(f'autoencoder/{file_path}_scaler.pkl', buffer)      
        elif (save=="local"): 
          # Save locally
          joblib.dump(scaler,f'{file_path}_scaler.pkl')
      else:
        pass
    else:
      if (save is not None):
        if (save=="BucketFS"):
          # Load from BucketFS
          data = client.download(f'autoencoder/{file_path}_scaler.pkl')
          scaler = joblib.load(data)
        elif (save=="local"):
          # Load locally
          scaler = joblib.load(f'{file_path}_scaler.pkl')
      else: 
        pass
      input_df_scaled = scaler.transform(input_df[continous_columns])
    input_df[continous_columns] = input_df_scaled

    # Preprocess categorical columns
    if (is_train == True):
      input_df_encoded = onehotencoder.fit_transform(input_df[categorical_columns])
      if (save is not None):  
        if (save=="BucketFS"):
          # Save to BucketFS
          buffer = io.BytesIO()
          joblib.dump(onehotencoder,buffer)
          client.upload(f'autoencoder/{file_path}_encoder.pkl', buffer)     
        elif (save=="local"):
          # Save locally
          joblib.dump(scaler,f'{file_path}_encoder.pkl')
      else:
        pass
    else:
      if (save is not None):
        if (save=="BucketFS"):
          # Load from BucketFS
          data = client.download(f'autoencoder/{file_path}_encoder.pkl')
          scaler = joblib.load(data)
        elif (save=="local"):
          # Load locally
          scaler = joblib.load(f'{file_path}_encoder.pkl')
      else:
        pass
      input_df_encoded = onehotencoder.transform(input_df[categorical_columns])
    input_df_encoded_part = pd.DataFrame(input_df_encoded, columns=onehotencoder.get_feature_names_out(categorical_columns),index=input_df.index)
    input_df = pd.concat([input_df,input_df_encoded_part],axis=1)
    input_df.drop(columns=categorical_columns, inplace=True)

    return input_df

class BucketFS_client():
    def __init__(self):
        """
         @brief Provide client API to interact with BucketFS
        """
        self.url = "http://172.18.0.2:6583"
        self.cred = {"default":{"username":"w","password":"write"}}
        self.bucketfs = Service(self.url,self.cred)
        self.bucket = self.bucketfs["default"]  
    
    def upload(self,file_path,buffer):
        """
         @brief Upload file to BucketFS
         @param file_path: Full path and file name in BucketFS
         @param buffer: BytesIO object to write to BucketFS as bytes
        """
        buffer.seek(0)             
        self.bucket.upload(file_path, buffer)

    def download(self,file_path):
        """
         @brief Upload file to BucketFS
         @param file_path: Full path and file name in BucketFS
        """
        data = data = io.BytesIO(as_bytes(self.bucket.download(file_path)))
        return data

class Autoencoder(nn.Module):
    
    def __init__(self, layers, dropout):
        """
         @brief Initialize Autoencoder with given layer sizes and dropout. This is the base constructor for Autoencoder. You can override it in your subclass if you want to customize the layers.
         @param layers: List of size of layers to use
         @param dropout: List of ( drop_layer, drop_chance )
        """
        super(Autoencoder, self).__init__()
        self.layer_sizes = layers
        self.num_layers = len(layers)

        # Encoder layers
        encoder_layers = []
        for i in range(self.num_layers - 1):
            encoder_layers.append(nn.Linear(layers[i], layers[i + 1]))
            encoder_layers.append(nn.ReLU())
            for drop_layer, drop_chance in dropout:
                if i == drop_layer:
                    encoder_layers.append(nn.Dropout(drop_chance))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        for i in range(self.num_layers - 1, 0, -1):
            decoder_layers.append(nn.Linear(layers[i], layers[i - 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(layers[0], layers[0]))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def build_autoencoder(layers,dropout,learning_rate=1e-3,weight_decay=1e-5,load_method=None,weight_path=None):
    """
     @brief Build autoencoder encoder decoder and optimizer.
     @param layers: A list specifying the number of layers and their respective size
     @param dropout: A list of tupple specifying dropout layers position and their respective dropout chance
     @param learning_rate:
     @param weight_decay:  
     @param load_method: Weight loading method. Can be "BucketFS" or "local". Disabled by default
     @param weight_path: Path of pretrained weight file
    """
    autoencoder = Autoencoder(layers,dropout)
    if (load_method is not None):
        assert weight_path is not None, "weight_path must be declared if load_method is specified."
    if (weight_path is not None):
        assert load_method is not None, "load_method must be declared if weight path is specified."

    if (load_method=="BucketFS"):
        # Load weight from BuckeFS
        client = BucketFS_client()
        weight = client.download(weight_path)
    elif(load_method=="local"):
        # Load weight by local file
        with open(weight_path, 'rb') as file:
            weight = io.BytesIO(file.read())

    if (load_method is not None):
        autoencoder.load_state_dict(torch.load(weight))

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return autoencoder, encoder, decoder, optimizer

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

def train(autoencoder,num_epochs,batch_size,patience,layers,train_loader,val_loader,continous_columns,categorical_columns,onehotencoder,scaler,optimizer,scheduler,device,save=None):
    """
     @brief Autoencoder trainer
     @param autoencoder: Autoencoder object
     @param num_epochs: Number of training epochs 
     @param batch_size: Traning batch size
     @param patience: Number of epochs to wait before stopping the training process if validation loss does not improve
     @param layers: A list specifying sizes of network layers
     @param train_loader: Dataloader object containing train dataset
     @param val_loader: Dataloader object containing validation dataset
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param onehotencoder: Onehot encoder object
     @param scaler: Scaler object
     @param optimizer: Optimizer object
     @param scheduler: Scheduler object
     @param device: Can be "cpu" or "cuda"
     @param save: Enable saving training weight. Can be "BucketFS" or a directory path. Default None.
    """
    best_loss = float('inf')
    best_state_dict = None

    # Training loop
    for epoch in range(num_epochs):
        train_progress = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}], training progress', position=0, leave=True)

        running_loss = 0.0
        running_sample_count = 0.0
        for inputs, _  in train_progress:
            # Forward pass
            inputs = inputs.to(device)
            outputs = autoencoder(inputs)

            loss = loss_CEMSE(inputs, outputs, continous_columns, categorical_columns, onehotencoder, scaler)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*batch_size
            running_sample_count += inputs.shape[0]

        average_loss = running_loss / running_sample_count      # Final loss: multiply by batch size then averaged over all samples
        train_progress.set_postfix({'Training Loss': average_loss})
        train_progress.update()
        train_progress.close()

        # Calculate validation loss
        val_progress = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{num_epochs}], validation progress', position=0, leave=True)

        val_running_loss = 0.0
        val_running_sample_count = 0.0
        for val_inputs, _ in val_progress:
            val_inputs = val_inputs.to(device)
            val_outputs = autoencoder(val_inputs)

            val_loss = loss_CEMSE(val_inputs, val_outputs, continous_columns, categorical_columns, onehotencoder, scaler)

            val_running_loss += val_loss.item()*batch_size
            val_running_sample_count += val_inputs.shape[0]

        val_avg_loss = val_running_loss / val_running_sample_count
        val_progress.set_postfix({'Validation Loss': val_avg_loss})
        val_progress.update()
        val_progress.close()

        # Check if validation loss has improved
        if val_avg_loss < best_loss - 0.0001:
            best_loss = val_avg_loss
            best_state_dict = autoencoder.state_dict()
            counter = 0
        else:
            counter += 1

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.8f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_avg_loss:.8f}")

        # Update the learning rate
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}]: Learning rate = {scheduler.get_last_lr()}\n")

        # Early stopping condition
        if counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break
        train_progress.close()
     
    # Save training weight 
    autoencoder.load_state_dict(best_state_dict)
    layers_str = '_'.join(str(item) for item in layers)
    file_name = f'autoencoder_{layers_str}.pth'
    if (save is not None): 
        if (save=="BucketFS"):   
            buffer = io.BytesIO()
            torch.save(autoencoder.state_dict(), buffer)
            client = BucketFS_client()
            client.upload(f'autoencoder/{file_name}', buffer)
        else:
            file_path = os.path.abspath(os.path.join(save, file_name))
            torch.save(autoencoder.state_dict(), file_path)
            print(f'Saved weight to {file_path}')
    else:
        pass

def anonymize(encoder,test_df,test_loader,batch_size,device):
    """
     @brief Data anonymizing using only the encoder
     @param encoder: Encoder object
     @param test_df: Test set dataFrame
     @param test_loader: Dataloader object containing test dataset
     @param batch_size: Anonymizing batch size
     @param device: can be "cpu" or "cuda"
    """
    encoder.eval()
    anonymize_progress = tqdm(test_loader, desc=f'Anonymize progress', position=0, leave=True)

    anonymized_outputs = torch.empty(0).to(device)
    with torch.no_grad():
        for inputs,_ in anonymize_progress:
            inputs = inputs.to(device)
            outputs = encoder(inputs)
            anonymized_outputs = torch.cat((anonymized_outputs,outputs),dim=0)
    
    anonymized_data = pd.DataFrame(anonymized_outputs.detach().cpu().numpy(),index=test_df.index[:(test_df.shape[0] // batch_size) * batch_size])
    return anonymized_data

def clean(autoencoder,test_df,test_loader,batch_size,continous_columns,categorical_columns,og_columns,onehotencoder,scaler,device):
    """
     @brief Data cleaning using the whole autoencoder
     @param autoencoder: Autoencoder object
     @param test_df: Test set dataframe
     @param test_loader: Dataloader object containing test dataset
     @param batch_size: Cleaning batch size 
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param og_columns: A list of original columns order 
     @param onehotencoder: Onehot encoder object
     @param scaler: Scaler object
     @param device: can be "cpu" or "cuda"
    """
    autoencoder.eval()

    clean_progress = tqdm(test_loader, desc=f'Clean progress', position=0, leave=True)
    clean_outputs = torch.empty(0).to(device)
    clean_loss = torch.empty(0).to(device)
    with torch.no_grad():
        for inputs,_ in clean_progress:
            inputs = inputs.to(device)
            outputs = autoencoder(inputs)
            outputs_con = outputs[:,:len(continous_columns)]
            outputs_cat = outputs[:,len(continous_columns):]
            outputs_cat = argmax(outputs_cat, onehotencoder, continous_columns, categorical_columns, device)
            outputs_final = torch.cat((outputs_con,outputs_cat),dim=1)
            clean_outputs = torch.cat((clean_outputs,outputs_final),dim=0)

            loss = torch.mean(torch.abs(outputs_final-inputs),dim=1)

            clean_loss = torch.cat((clean_loss,loss),dim=0)
    avg_loss = torch.mean(clean_loss)
    print(f'\nMAE: {avg_loss:.8f}')

    clean_data = pd.DataFrame(clean_outputs.detach().cpu().numpy(),columns=test_df.columns,index=test_df.index[:(test_df.shape[0] // batch_size) * batch_size])
    decoded_cat_cols = pd.DataFrame(onehotencoder.inverse_transform(clean_data.iloc[:,len(continous_columns):]),index=clean_data.index,columns=categorical_columns)
    decoded_con_cols = pd.DataFrame(scaler.inverse_transform(clean_data.iloc[:,:len(continous_columns)]),index=clean_data.index,columns=continous_columns).round(0)
    clean_data = pd.concat([decoded_con_cols,decoded_cat_cols],axis=1).reindex(columns=og_columns)
    return clean_data

def outlier_dectection():
    pass
