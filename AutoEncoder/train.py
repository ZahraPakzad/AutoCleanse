import os
import io
import torch
from exasol.bucketfs import Service
from tqdm import tqdm
from AutoEncoder.loss_model import loss_CEMSE
from AutoEncoder.bucketfs_client import BucketFS_client

def train(model,num_epochs,batch_size,patience,layers,train_loader,val_loader,onehotencoder,scaler, \
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
