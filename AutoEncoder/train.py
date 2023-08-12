import os
import io
import torch
from exasol.bucketfs import Service
from tqdm import tqdm
from AutoEncoder.loss_model import loss_CEMSE

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
            buffer.seek(0)

            url = "http://172.18.0.2:6583"
            cred ={"default":{"username":"w","password":"write"}}
            bucketfs = Service(url,cred)
            bucket = bucketfs["default"]          
            bucket.upload(f"autoencoder/{file_name}", buffer)
        else:
            file_path = os.path.abspath(os.path.join(save, file_name))
            torch.save(autoencoder.state_dict(), file_path)
            print(f'Saved weight to {file_path}')
    else:
        pass
