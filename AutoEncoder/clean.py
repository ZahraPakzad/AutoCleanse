import torch
import pandas as pd
from tqdm import tqdm
# from AutoEncoder.utils import argmax,softmax

from utils import argmax,softmax

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
    autoencoder.to(device)
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