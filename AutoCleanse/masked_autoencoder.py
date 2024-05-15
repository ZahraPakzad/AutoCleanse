import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import *
# from AutoCleanse.bucketfs_client import bucketfs_client
# from AutoCleanse.utils import *
# from AutoCleanse.loss_model import loss_CEMSE

from bucketfs_client import BucketFSClient
from utils import *
from loss_model import loss_CEMSE
from autoencoder import *

from torch.optim.lr_scheduler import StepLR

class MaskedAutoencoder(nn.Module):
    
    def __init__(self, layers, batch_norm, dropout_enc=None, dropout_dec=None, l1_strength=0.0, l2_strength=0.0,
                 learning_rate=1e-3, weight_decay=0, mask_ratio=0.1):
        super(MaskedAutoencoder, self).__init__()
        self.layers = layers
        self.num_layers = len(layers)
        self.wlc = None 
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.best_state_dict = None
        self.mask_ratio = mask_ratio  # Store mask ratio
        
        # Encoder layers
        encoder_layers = []
        for i in range(self.num_layers - 1):
            encoder_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if batch_norm == True:
                encoder_layers.append(nn.BatchNorm1d(layers[i + 1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers[-1].register_forward_hook(self.add_regularization_hook) 
            if dropout_enc is not None:
                for drop_layer, drop_chance in dropout_enc:
                    if i == drop_layer:
                        encoder_layers.append(nn.Dropout(drop_chance))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        for i in range(self.num_layers - 1, 0, -1):
            decoder_layers.append(nn.Linear(layers[i], layers[i - 1]))
            if batch_norm == True:                
                decoder_layers.append(nn.BatchNorm1d(layers[i - 1]))
            decoder_layers.append(nn.ReLU())
            if dropout_dec is not None:
                for drop_layer, drop_chance in dropout_dec:
                    if (i == self.num_layers - 1 - drop_layer):
                        decoder_layers.append(nn.Dropout(drop_chance))
            
        decoder_layers.append(nn.Linear(layers[0], layers[0]))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=25, gamma=0.1)

    def add_regularization_hook(self, module, input, output):
        l1_reg = self.l1_strength * F.l1_loss(output, torch.zeros_like(output))
        l2_reg = self.l2_strength * F.mse_loss(output, torch.zeros_like(output))
        module.register_forward_hook(None)  
        module._forward_hooks.clear()
        return output + l1_reg + l2_reg 
    
    def forward(self, x, inference = False):  # todo: fix name to eval method instead of inference
        # Generate mask with the specified ratio
        if inference== False: # for training, apply random masking to the data
            mask = torch.rand_like(x) > self.mask_ratio
            encoded = self.encoder(x * mask)  # Apply masking to the input
        else: 
            # The method of replacing nan values using a Masked Auto Encoder is in mind. 
            # When used during inference (cleaning data), a mask of the nan values should be made,
            # so that the model would predict those values. 
            # what was tried: making a mapping of the nan values using the isnan() function of torch.
            # It was assumed that the True/False map would be multiplied with the input to "mask/remove" those values during encodin (just like what is being done in training).
            # The result appeared to return nan values when a boolean value (True) was multiplied with it. 
            # Therefore, a replacement of the nan value with "0.0" (numerical, not string) was seemed as workaround which gave us the same result. 
            # (replacement with "0.0" was chosen as it was also being done in the training phase in the lines above.)
            # However, this is not the best solution. As there are also values of 0.0 in the dataset and the model
            # would assume the nans and the "0.0"s would have the same meaning. 
            # p.s: The 'mask' value is only here for similarity of the code in training and inferencing. 
            # It does not provide any extra logic, as the actual 'masking' is being done in conversion of nan values to '0.0's. 

            # Todo: choosing a better method of masking than having a mask with "0.0" values. 
            # The link below was checked as an implementation of the code in the main MaskedAutoEncoder papers. 
            # What was understood is that an indexing system is being used in a way that in the encoding phase, some pixel values are being removed from the 
            # image, while having maintained their index. The indexes are added back for decoding as a placeholder to be filled. 
            # https://github.com/IcarusWizard/MAE/blob/main/model.py
             
            mask = ~torch.isnan(x) # mapping of the nan values. 
                                    # They are inverted as the False values multiplied with other non-nan values turns them into 0.0s. 
            x = torch.nan_to_num(x, nan=100.0) # todo: use a specific big number
            encoded = self.encoder(x * mask)  # Apply masking to the input
            # encoded = self.encoder(x) 

        decoded = self.decoder(encoded)
        return decoded


    def train_model(self,num_epochs,batch_size,patience,train_loader,val_loader,categories, \
                    device,continous_columns,categorical_columns,wlc=(1,1)):
        """
        Train the model using the specified parameters and data loaders.

        Args:
            num_epochs (int): The number of epochs for training.
            batch_size (int): The batch size for the data loaders.
            patience (int): The number of epochs to wait for improvement before stopping training.
            train_loader (DataLoader): The data loader for the training set.
            val_loader (DataLoader): The data loader for the validation set.
            categories (list): The list of category names.
            device (str): The device to use for training (e.g., 'cpu', 'cuda').
            continous_columns (list): The list of names of the continuous columns.
            categorical_columns (list): The list of names of the categorical columns.
            wlc (tuple, optional): The weighted loss coefficients for CE and MSE losses. Defaults to (1, 1).

        Returns:
            None
        """
        
        self.wlc =  wlc
        best_loss = float('inf')
        self.to(device)
        counter = 0
        # Training loop
        for epoch in range(num_epochs):
            train_progress = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}], Training Progress', position=0, leave=True)
            print("train progress: ")
            print(train_progress)
            running_loss = 0.0
            running_loss_comp = 0.0
            running_CEloss = 0.0
            running_MSEloss = 0.0
            running_sample_count = 0.0
            for inputs, _  in train_progress:
                # Forward pass
                inputs = inputs.to(device)
                outputs= self(inputs)
                CEloss,MSEloss = loss_CEMSE(inputs, outputs, categories, continous_columns, categorical_columns)
                if (len(categorical_columns)==0):
                    loss = wlc[0]*CEloss + wlc[1]*MSEloss 
                    loss_comp = MSEloss
                    
                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()*batch_size
                    running_loss_comp += loss_comp.item()*batch_size
                    running_MSEloss += MSEloss.item()*batch_size
                    running_sample_count += inputs.shape[0]
                else:
                    loss = wlc[0]*CEloss + wlc[1]*MSEloss 
                    loss_comp = CEloss + MSEloss 

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()*batch_size
                    running_loss_comp += loss_comp.item()*batch_size
                    running_CEloss += CEloss.item()*batch_size
                    running_MSEloss += MSEloss.item()*batch_size
                    running_sample_count += inputs.shape[0]

            average_loss = running_loss / running_sample_count      # Final loss: multiply by batch size then averaged over all samples
            average_loss_comp = running_loss_comp / running_sample_count
            average_CEloss = running_CEloss / running_sample_count
            average_MSEloss = running_MSEloss / running_sample_count
            train_progress.set_postfix({"Training Loss": average_loss})
            train_progress.update()
            train_progress.close()

            # Calculate validation loss
            val_progress = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{num_epochs}], Validation Progress', position=0, leave=True)

            val_running_loss = 0.0
            val_running_loss_comp = 0.0
            val_running_CEloss = 0.0
            val_running_MSEloss = 0.0
            val_running_sample_count = 0.0
        
            for val_inputs, _ in val_progress:
                val_inputs = val_inputs.to(device)
                val_outputs= self(val_inputs) # here: gives the value to the model and expects its output

                val_CEloss,val_MSEloss = loss_CEMSE(val_inputs, val_outputs, categories, continous_columns, categorical_columns)
                if (len(categorical_columns)==0):
                    val_loss = wlc[0]*val_CEloss + wlc[1]*val_MSEloss
                    val_loss_comp = val_CEloss + val_MSEloss 

                    val_running_loss += val_loss.item()*batch_size
                    val_running_loss_comp += val_loss_comp.item()*batch_size
                    val_running_MSEloss += val_MSEloss.item()*batch_size
                    val_running_sample_count += val_inputs.shape[0]
                else:
                    val_loss = wlc[0]*val_CEloss + wlc[1]*val_MSEloss
                    val_loss_comp = val_CEloss + val_MSEloss 

                    val_running_loss += val_loss.item()*batch_size
                    val_running_loss_comp += val_loss_comp.item()*batch_size
                    val_running_CEloss += val_CEloss.item()*batch_size
                    val_running_MSEloss += val_MSEloss.item()*batch_size
                    val_running_sample_count += val_inputs.shape[0]


            val_avg_loss = val_running_loss / val_running_sample_count
            val_avg_loss_comp = val_running_loss_comp / val_running_sample_count
            val_average_CEloss = val_running_CEloss / val_running_sample_count
            val_average_MSEloss = val_running_MSEloss / val_running_sample_count
            val_progress.set_postfix({"Validation Loss": val_avg_loss})
            val_progress.update()
            val_progress.close()

            # Check if validation loss has improved
            if val_avg_loss < best_loss - 0.001:
                best_loss = val_avg_loss
                self.best_state_dict = self.state_dict()
                counter = 0
            else:
                counter += 1

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.8f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_avg_loss:.8f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Training CE Loss: {average_CEloss:.8f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation CE Loss: {val_average_CEloss:.8f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Training MSE Loss: {average_MSEloss:.8f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation MSE Loss: {val_average_MSEloss:.8f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss Comp: {average_loss_comp:.8f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss Comp: {val_avg_loss_comp:.8f}")

            # Update the learning rate
            self.scheduler.step()
            print(f"Epoch [{epoch+1}/{num_epochs}]: Learning Rate = {self.scheduler.get_last_lr()}\n")

            # Early stopping condition
            if counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break
            train_progress.close()
            val_progress.close()
        
    def save(self,location,name=None):
        self.load_state_dict(self.best_state_dict)
        if (name is None):
            layers_str = '_'.join(str(item) for item in self.layers) 
            wlc_str = str(self.wlc)
            name = f'autoencoder_{layers_str}_{wlc_str}.pth'
        else:
            name = f'autoencoder_{name}.pth'
        if (location=="bucketfs"):
            buffer = io.BytesIO()
            torch.save(self.state_dict(), buffer)
            try:
                bucketfs_client().upload(f'autoencoder/{name}',buffer)
            except Exception as e:
                raise RuntimeError(f"Failed saving {name} to BucketFS") from e
            print(f'Saved weight to default/autoencoder/{name}')
        elif (location=="local"):
            try:
                torch.save(self.state_dict(), name)
            except Exception as e:
                raise RuntimeError(f"Failed saving {name} to local") from e
            print(f'Saved weight to {name}')

    def load(self,location,name=None):
        weight = None 
        name = f"autoencoder_{name}.pth"           
        if (location=="bucketfs"):
            try:
                weight = bucketfs_client().download(f'autoencoder/{name}')
            except Exception as e:
                raise RuntimeError(f"Failed loading {name} from BucketFS") from e
            print(f'Loaded weight from default/autoencoder/{name}')
        elif (location=="local"):
            try:
                with open(name, 'rb') as file:
                    weight = io.BytesIO(file.read())
            except Exception as e:
                raise RuntimeError(f"Failed loading {name} from local") from e
            print(f'Loaded weight from {name}')
        self.load_state_dict(torch.load(weight))

    def clean(self,dirty_loader,df,batch_size,onehotencoder,scaler,device,\
              og_columns,continous_columns=None,categorical_columns=None,test_loader=None, is_mae=False):
        """
        Clean the test data using the trained model and return the cleaned data.

        Parameters:
            df (DataFrame): The dataframe to be cleaned.
            dirty_loader (DataLoader): The DataLoader for the dirty data. Dirty data is data that actually need to be cleaned.
            test_loader (DataLoader): The DataLoader for the test data. Test data is the original clean version of dirty data. 
                                      This is only used to test the performance of the model agaisnt artificial dirty data.
            batch_size (int): The batch size for processing the data.
            onehotencoder (OneHotEncoder): The one-hot encoder for categorical columns.
            scaler (Scaler): The scaler for continuous columns.
            device (str): The device to be used for processing (e.g., 'cpu' or 'cuda').
            og_columns (List): The original columns of the test dataset.
            continous_columns (List, optional): The list of continuous columns. Defaults to None.
            categorical_columns (List, optional): The list of categorical columns. Defaults to None.

        Returns:
            clean_data (DataFrame): The cleaned test data.  
        """
        
        self.eval() 
        self.to(device)
        clean_outputs = torch.empty(0, device=device)
        if (test_loader is not None):
            clean_progress = tqdm(zip(dirty_loader,test_loader), desc=f'Clean progress', total=len(dirty_loader), position=0, leave=True)
            MAE = torch.empty(0, device=device)
            MSE = torch.empty(0, device=device)
            with torch.no_grad():
                for batch_dirty,batch_test in clean_progress:
                    inputs_dirty,_ = batch_dirty
                    inputs_test,_ = batch_test
                    inputs_dirty = inputs_dirty.to(device)
                    inputs_test = inputs_test.to(device)
                    if is_mae == True:
                        outputs = self(inputs_dirty, inference = True)
                    else: 
                        outputs = self(inputs_dirty)
                    
                    # outputs = self(inputs_dirty)
                    outputs_final = torch.empty(0, device=device)                
                    if (len(continous_columns)!=0 and len(categorical_columns)!=0):
                        outputs_con = outputs[:,:len(continous_columns)]
                        outputs_cat = outputs[:,len(continous_columns):]
                        outputs_cat = argmax(outputs_cat, onehotencoder, continous_columns, categorical_columns, device)
                        outputs_final = torch.cat((outputs_con,outputs_cat),dim=1)
                    elif (len(continous_columns)==0):              
                        outputs_final = argmax(outputs, onehotencoder, continous_columns, categorical_columns, device)
                    elif (len(categorical_columns)==0):
                        outputs_final = outputs

                    clean_outputs = torch.cat((clean_outputs,outputs_final),dim=0)

                    MAEloss = torch.unsqueeze(F.l1_loss(outputs_final,inputs_test),dim=0)
                    MSEloss = torch.unsqueeze(F.mse_loss(outputs_final,inputs_test),dim=0)

                    MAE = torch.cat((MAE,MAEloss),dim=0)
                    MSE = torch.cat((MSE,MSEloss),dim=0)
            MAEavg = torch.mean(MAE)
            MSEavg = torch.mean(MSE)
            print(f'\nMAE: {MAEavg:.8f}')
            print(f'\nMSE: {MSEavg:.8f}')
        else:
            clean_progress = tqdm(dirty_loader, desc=f'Clean progress', total=len(dirty_loader), position=0, leave=True)            
            with torch.no_grad():
                for inputs,_ in clean_progress:
                    inputs = inputs.to(device)
                    if is_mae == True:
                        outputs = self(inputs, inference = True)
                    else: 
                        outputs = self(inputs)
                    
                    if (len(continous_columns)!=0 and len(categorical_columns)!=0):
                        outputs_con = outputs[:,:len(continous_columns)]
                        outputs_cat = outputs[:,len(continous_columns):]
                        outputs_cat = argmax(outputs_cat, onehotencoder, continous_columns, categorical_columns, device)
                        outputs_final = torch.cat((outputs_con,outputs_cat),dim=1)
                    elif (len(continous_columns)==0):                
                        outputs_final = argmax(outputs, onehotencoder, continous_columns, categorical_columns, device)
                    elif (len(categorical_columns)==0):
                        outputs_final = outputs
                    clean_outputs = torch.cat((clean_outputs,outputs_final),dim=0)

        clean_data = pd.DataFrame(clean_outputs.detach().cpu().numpy(),columns=df.columns,index=df.index[:(df.shape[0] // batch_size) * batch_size])
        # removing the inverse scaling of the values, as the scaled values of the dirty dataset are still needed as the input to the main auto encoder.
        # if (len(continous_columns)!=0 and len(categorical_columns)!=0):
        #     decoded_cat_cols = pd.DataFrame(onehotencoder.inverse_transform(clean_data.iloc[:,len(continous_columns):]),index=clean_data.index,columns=categorical_columns)
        #     decoded_con_cols = pd.DataFrame(scaler.inverse_transform(clean_data.iloc[:,:len(continous_columns)]),index=clean_data.index,columns=continous_columns).round(0)
        #     clean_data = pd.concat([decoded_con_cols,decoded_cat_cols],axis=1).reindex(columns=og_columns)
        # elif (len(continous_columns)==0):
        #     clean_data = pd.DataFrame(onehotencoder.inverse_transform(clean_data),index=clean_data.index,columns=categorical_columns)
        # elif (len(categorical_columns)==0):
        #     clean_data = pd.DataFrame(scaler.inverse_transform(clean_data),index=clean_data.index,columns=continous_columns).round(0)
        
        return clean_data

    def anonymize(self,df,data_loader,batch_size,device):
        """
        Anonymizes input data using the encoder model and returns the anonymized data as a DataFrame.

        Args:
            test_df (DataFrame): The input test data as a DataFrame.
            test_loader (DataLoader): The data loader for the test data.
            batch_size (int): The batch size for processing the data.
            device (str): The device to be used for processing.

        Returns:
            DataFrame: The anonymized data as a DataFrame.
        """
        
        self.encoder.eval()
        self.encoder.to(device)
        anonymize_progress = tqdm(data_loader, desc=f'Anonymize progress', position=0, leave=True)

        anonymized_outputs = torch.empty(0).to(device)
        with torch.no_grad():
            for inputs,_ in anonymize_progress:
                inputs = inputs.to(device)
                outputs = self.encoder(inputs)
                anonymized_outputs = torch.cat((anonymized_outputs,outputs),dim=0)
        
        anonymized_data = pd.DataFrame(anonymized_outputs.detach().cpu().numpy(),index=df.index[:(df.shape[0] // batch_size) * batch_size])
        return anonymized_data


    def outlier_dectection():
        pass
        


