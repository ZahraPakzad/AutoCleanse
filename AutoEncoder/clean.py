import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from AutoEncoder.utils import argmax,softmax

def clean(autoencoder,test_df,test_loader,test_loader_og,batch_size,continous_columns,categorical_columns,og_columns,onehotencoder,scaler,device):
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
    autoencoder.to(device)
    clean_progress = tqdm(zip(test_loader,test_loader_og), desc=f'Clean progress', total=len(test_loader), position=0, leave=True)
    clean_outputs = torch.empty(0, device=device)
    MAE = torch.empty(0, device=device)
    MSE = torch.empty(0, device=device)
    with torch.no_grad():
        for batch_test,batch_test_og in clean_progress:
            inputs,_ = batch_test
            inputs_og,_ = batch_test_og
            inputs = inputs.to(device)
            inputs_og = inputs_og.to(device)

            outputs = autoencoder(inputs)
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

            MAEloss = torch.unsqueeze(F.l1_loss(outputs_final,inputs_og),dim=0)
            MSEloss = torch.unsqueeze(F.mse_loss(outputs_final,inputs_og),dim=0)

            MAE = torch.cat((MAE,MAEloss),dim=0)
            MSE = torch.cat((MSE,MSEloss),dim=0)
    MAEavg = torch.mean(MAE)
    MSEavg = torch.mean(MSE)
    print(f'\nMAE: {MAEavg:.8f}')
    print(f'\nMSE: {MSEavg:.8f}')

    clean_data = pd.DataFrame(clean_outputs.detach().cpu().numpy(),columns=test_df.columns,index=test_df.index[:(test_df.shape[0] // batch_size) * batch_size])
    if (len(continous_columns)!=0 and len(categorical_columns)!=0):
        decoded_cat_cols = pd.DataFrame(onehotencoder.inverse_transform(clean_data.iloc[:,len(continous_columns):]),index=clean_data.index,columns=categorical_columns)
        decoded_con_cols = pd.DataFrame(scaler.inverse_transform(clean_data.iloc[:,:len(continous_columns)]),index=clean_data.index,columns=continous_columns).round(0)
        clean_data = pd.concat([decoded_con_cols,decoded_cat_cols],axis=1).reindex(columns=og_columns)
    elif (len(continous_columns)==0):
        clean_data = pd.DataFrame(onehotencoder.inverse_transform(clean_data),index=clean_data.index,columns=categorical_columns)
    elif (len(categorical_columns)==0):
        clean_data = pd.DataFrame(scaler.inverse_transform(clean_data),index=clean_data.index,columns=continous_columns).round(0)
    
    return clean_data

def outlier_dectection():
    pass