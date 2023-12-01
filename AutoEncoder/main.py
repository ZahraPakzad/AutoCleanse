import torch 
import pandas as pd
import torch.nn as nn
import time
import io
import joblib
import argparse

from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from tqdm import tqdm
from tabulate import tabulate

from AutoEncoder.utils import *
from AutoEncoder.dataloader import PlainDataset, DataLoader
from AutoEncoder.autoencoder import *
from AutoEncoder.loss_model import loss_CEMSE
from AutoEncoder.preprocessor import dataSplitter,dataPreprocessor
from AutoEncoder.anonymize import anonymize

start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-l','--layers', type=str, help='Layer configuration')
parser.add_argument('-w','--wlc', type=str, help='Weighted loss coefficient')
args = parser.parse_args()

# Loading dataset
df = pd.read_csv('~/dataset/categorical/adult.csv').drop(columns=['fnlwgt','capital.gain','capital.loss','income'])
continous_columns = ['age','hours.per.week']
categorical_columns = ['workclass','education','education.num','marital.status','occupation','relationship','race','sex','native.country'] 
# continous_columns = X.select_dtypes(include=['int64', 'float64']).columns     #@TODO: make these global
# categorical_columns = X.select_dtypes(include=['object', 'bool']).columns
og_columns = df.columns.to_list()
df = df[continous_columns+categorical_columns]

X_train,X_val,X_test_og = dataSplitter(input_df=df,
                                    train_ratio=0.7,
                                    val_ratio=0.15,
                                    test_ratio=0.15,
                                    random_seed=42)
X_test = replace_with_nan(X_test_og,0,42)

# layers = string2list(args.layers) #@TODO: just a hack to name saved files between runs. Kinda ugly. See model declaration below to know why.
# wlc = string2tupple(args.wlc)
layers = [1024,128]
wlc = (1,5)

X_train,scaler,onehotencoder = dataPreprocessor(
                        input_df=X_train,
                        is_train=True,             
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        load_method="local",
                        layers=layers)

X_val,scaler,onehotencoder = dataPreprocessor(
                        input_df=X_val,    
                        is_train=False,         
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        load_method="local",
                        layers=layers)                            

X_test,scaler,onehotencoder = dataPreprocessor(
                        input_df=X_test,   
                        is_train=False,
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        load_method="local",
                        layers=layers)     

X_test_og,scaler,onehotencoder = dataPreprocessor(
                        input_df=X_test_og,   
                        is_train=False,
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        load_method="local",
                        layers=layers)                            

# Create dataloader
train_dataset = PlainDataset(X_train)
val_dataset = PlainDataset(X_val)
test_dataset = PlainDataset(X_test)
test_dataset_og = PlainDataset(X_test_og)

def custom_collate_fn(batch):
    tensor_data = torch.stack([item[0] for item in batch])
    indices = [item[1] for item in batch]
    return tensor_data, indices

torch.manual_seed(42) #@TODO: random seed 
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
test_loader_og = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)

# Declaring model
layers = [X_train.shape[1]]+layers                                  
autoencoder, encoder, decoder, optimizer = Autoencoder.build_model(layers=layers,dropout_enc=[(0,0.0)],dropout_dec=[(0,0.1)], batch_norm=True, \
                                                             learning_rate=1e-4,weight_decay=1e-5,l1_strength=1e-5,l2_strength=1e-5, \
                                                             load_method="local",weight_path="/home/tung/development/AutoEncoder/autoencoder_1024_128_(1, 5).pth")

scheduler = StepLR(optimizer, step_size=25, gamma=0.1)   # LR*0.1 every 30 epochs
# summary(autoencoder.to(device),torch.tensor(X_train.values).float().to(device).shape[1:])

# autoencoder.train(
#       patience=10,
#       num_epochs=150,
#       batch_size=batch_size,
#       layers=layers,
#       train_loader=train_loader,
#       val_loader=val_loader,
#       continous_columns=continous_columns, 
#       categorical_columns=categorical_columns, 
#       onehotencoder=onehotencoder, 
#       scaler=scaler,
#       optimizer=optimizer,
#       scheduler=scheduler,
#       device=device,
#       loss_ratio=wlc,
#       save="local")

cleaned_data = autoencoder.clean(test_loader=test_loader,
                                test_loader_og=test_loader_og,
                                test_df=X_test,
                                batch_size=batch_size,
                                continous_columns=continous_columns, 
                                categorical_columns=categorical_columns, 
                                og_columns=og_columns,
                                onehotencoder=onehotencoder, 
                                scaler=scaler,
                                device=device)                    

# anonymized_data = anonymize(encoder=encoder,
#                             test_df=X_test,
#                             test_loader=test_loader,
#                             batch_size=batch_size,
#                             device=device) 

# print("\n")
print(tabulate(df.loc[[28296,28217,8054,4223,22723],og_columns],headers=og_columns,tablefmt="simple",maxcolwidths=[None, 4]))
print("\n")
print(tabulate(cleaned_data.loc[[28296,28217,8054,4223,22723]],headers=cleaned_data.columns.to_list(),tablefmt="simple",maxcolwidths=[None, 4]))
print("\n")
# print(tabulate(anonymized_data.round(decimals=4).iloc[:5,:32],headers=anonymized_data.columns.to_list(),tablefmt="simple",maxcolwidths=[None, 6]))
# print("\n")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# cleaned_data.to_csv('df_cleaned.csv',index=False)