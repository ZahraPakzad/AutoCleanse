import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
from AutoEncoder.bucketfs_client import BucketFS_client
from AutoEncoder.utils import generate_suffix
from exasol.bucketfs import Service
from tqdm import tqdm
from AutoEncoder.loss_model import loss_CEMSE
from AutoEncoder.bucketfs_client import BucketFS_client

class Autoencoder(nn.Module):
    
    def __init__(self, layers, batch_norm, dropout_enc=None, dropout_dec=None, l1_strength=0.0, l2_strength=0.0):
        """
         @brief Initialize Autoencoder with given layer sizes and dropout. This is the base constructor for Autoencoder. You can override it in your subclass if you want to customize the layers.
         @param layers: List of size of layers to use
         @param dropout: List of ( drop_layer, drop_chance )
        """
        super(Autoencoder, self).__init__()
        self.num_layers = len(layers)
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

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
                encoder_layers.append(nn.BatchNorm1d(layers[i - 1]))
            decoder_layers.append(nn.ReLU())
            if dropout_dec is not None:
                for drop_layer, drop_chance in dropout_dec:
                    if (i == self.num_layers - 1 - drop_layer):
                        decoder_layers.append(nn.Dropout(drop_chance))
            
        decoder_layers.append(nn.Linear(layers[0], layers[0]))
        self.decoder = nn.Sequential(*decoder_layers)

    def add_regularization_hook(self, module, input, output):
        l1_reg = self.l1_strength * F.l1_loss(output, torch.zeros_like(output))
        l2_reg = self.l2_strength * F.mse_loss(output, torch.zeros_like(output))
        module.register_forward_hook(None)  
        module._forward_hooks.clear()
        return output + l1_reg + l2_reg 

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    @classmethod
    def build_model(cls,layers,dropout_enc,dropout_dec,batch_norm,learning_rate=1e-3,weight_decay=0,l1_strength=0,l2_strength=0,load_method=None,weight_path=None):
        """
        @brief Build autoencoder encoder decoder and optimizer.
        @param layers: A list specifying the number of layers and their respective size
        @param dropout: A list of tupple specifying dropout layers position and their respective dropout chance
        @param learning_rate:
        @param weight_decay:  
        @param load_method: Weight loading method. Can be "BucketFS" or "local". Disabled by default
        """
        autoencoder = Autoencoder(layers=layers,
                                dropout_enc=dropout_enc,
                                dropout_dec=dropout_dec,
                                batch_norm=batch_norm,
                                l1_strength=l1_strength,
                                l2_strength=l2_strength)
        
        if (weight_path is None):
            weight_path = generate_suffix(layers,'autoencoder',load_method)

        if (load_method is not None):    
            if (load_method=="BucketFS"):
                # Load weight from BuckeFS
                client = BucketFS_client()
                weight = client.download(weight_path)
            elif(load_method=="local"):
                # Load weight by local file
                with open(weight_path, 'rb') as file:
                    weight = io.BytesIO(file.read())
            autoencoder.load_state_dict(torch.load(weight))

        encoder = autoencoder.encoder
        decoder = autoencoder.decoder
        optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return autoencoder, encoder, decoder, optimizer

    def train(self,model,num_epochs,batch_size,patience,layers,train_loader,val_loader,onehotencoder,scaler, \
          optimizer,scheduler,device,continous_columns,categorical_columns,loss_ratio=(1,1),save=None):
        """
        @brief Autoencoder trainer
        @param model: model object
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
        model.to(device)
        counter = 0
        # Training loop
        for epoch in range(num_epochs):
            train_progress = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}], Training Progress', position=0, leave=True)

            running_loss = 0.0
            running_loss_comp = 0.0
            running_CEloss = 0.0
            running_MSEloss = 0.0
            running_sample_count = 0.0
            for inputs, _  in train_progress:
                # Forward pass
                inputs = inputs.to(device)
                outputs = model(inputs)

                CEloss,MSEloss = loss_CEMSE(inputs, outputs, onehotencoder, scaler, continous_columns, categorical_columns)
                loss = loss_ratio[0]*CEloss + loss_ratio[1]*MSEloss
                loss_comp = CEloss + MSEloss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
                val_outputs = model(val_inputs)

                val_CEloss,val_MSEloss = loss_CEMSE(val_inputs, val_outputs, onehotencoder, scaler, continous_columns, categorical_columns)
                val_loss = loss_ratio[0]*val_CEloss + loss_ratio[1]*val_MSEloss
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
                best_state_dict = model.state_dict()
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
            scheduler.step()
            print(f"Epoch [{epoch+1}/{num_epochs}]: Learning Rate = {scheduler.get_last_lr()}\n")

            # Early stopping condition
            if counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break
            train_progress.close()
        
        # Save training weight 
        if (save is not None): 
            model.load_state_dict(best_state_dict)
            layers_str = '_'.join(str(item) for item in layers[1:]) #@TODO: file name hack
            file_name = f'autoencoder_{layers_str}_{loss_ratio}.pth'
            if (save=="BucketFS"):   
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                client = BucketFS_client()
                client.upload(f'autoencoder/{file_name}', buffer)
            elif (save=="local"):
                torch.save(model.state_dict(), file_name)
                print(f'Saved weight to {file_name}')
        else:
            pass
        
