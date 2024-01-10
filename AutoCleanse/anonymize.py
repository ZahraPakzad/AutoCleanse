import torch
import pandas as pd
from tqdm import tqdm

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
    encoder.to(device)
    anonymize_progress = tqdm(test_loader, desc=f'Anonymize progress', position=0, leave=True)

    anonymized_outputs = torch.empty(0).to(device)
    with torch.no_grad():
        for inputs,_ in anonymize_progress:
            inputs = inputs.to(device)
            outputs = encoder(inputs)
            anonymized_outputs = torch.cat((anonymized_outputs,outputs),dim=0)
    
    anonymized_data = pd.DataFrame(anonymized_outputs.detach().cpu().numpy(),index=test_df.index[:(test_df.shape[0] // batch_size) * batch_size])
    return anonymized_data
