import torch 
import pandas as pd
import torch.nn as nn
import time
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from tqdm import tqdm
from tabulate import tabulate

from utils import argmax, softmax
from dataloader import MyDataset, DataLoader
from autoencoder import build_autoencoder
from loss_model import loss_CEMSE
from preprocessor import dataSplitter,dataPreprocessor
from train import train
from clean import clean
from anonymize import anonymize

start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading dataset
df = pd.read_csv('~/dataset/categorical/adult.csv')
og_columns = df.columns.to_list()
continous_columns = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']
categorical_columns = ['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income']
df = df[continous_columns+categorical_columns]

scaler = StandardScaler()
onehotencoder = OneHotEncoder(sparse=False)

X_train,X_val,X_test = dataSplitter(input_df=df,
                                    train_ratio=0.7,
                                    val_ratio=0.15,
                                    test_ratio=0.15,
                                    random_seed=42)

X_train = dataPreprocessor(input_df=X_train,
                        is_train=True,             
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        scaler=scaler,
                        onehotencoder=onehotencoder,
                        save="local",
                        file_path="test")

X_val = dataPreprocessor(input_df=X_val,    
                        is_train=False,         
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        scaler=scaler,
                        onehotencoder=onehotencoder,
                        save="local",
                        file_path="test")      

X_test = dataPreprocessor(input_df=X_test,   
                        is_train=False,          
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        scaler=scaler,
                        onehotencoder=onehotencoder,
                        save="local",
                        file_path="test")                                         

# Create dataloader
train_dataset = MyDataset(X_train)
val_dataset = MyDataset(X_val)
test_dataset = MyDataset(X_test)

def custom_collate_fn(batch):
    tensor_data = torch.stack([item[0] for item in batch])
    indices = [item[1] for item in batch]
    return tensor_data, indices

torch.manual_seed(42) #@TODO: random seed 
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)

# Declaring model
layers = [X_train.shape[1],1000,500,35]

# weight_path="/home/tung/development/AutoEncoder/autoencoder_110_1000_500_35_gold.pth"
# autoencoder, encoder, decoder, optimizer = build_autoencoder(layers,dropout=[(0,0.1)],load_method="local",weight_path=weight_path)

weight_path="/autoencoder/autoencoder_110_1000_500_35.pth"
autoencoder, encoder, decoder, optimizer = build_autoencoder(layers,dropout=[(0,0.1)],load_method="BucketFS",weight_path=weight_path)

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)   # LR*0.1 every 30 epochs

autoencoder.to(device)
summary(autoencoder, (X_train.shape))

# train(autoencoder=autoencoder,
#       patience=15,
#       num_epochs=1,
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
#       save="BucketFS")

cleaned_data = clean(autoencoder=autoencoder,
                     test_loader=test_loader,
                     test_df=X_test,
                     batch_size=batch_size,
                     continous_columns=continous_columns, 
                     categorical_columns=categorical_columns, 
                     og_columns=og_columns,
                     onehotencoder=onehotencoder, 
                     scaler=scaler,
                     device=device)                    

anonymized_data = anonymize(encoder=encoder,
                            test_df=X_test,
                            test_loader=test_loader,
                            batch_size=batch_size,
                            device=device) 

print("\n")
print(tabulate(df.loc[[28296,28217,8054,4223,22723],og_columns],headers=og_columns,tablefmt="simple",maxcolwidths=[None, 4]))
print("\n")
print(tabulate(cleaned_data.head(),headers=cleaned_data.columns.to_list(),tablefmt="simple",maxcolwidths=[None, 4]))
print("\n")
print(tabulate(anonymized_data.round(decimals=4).iloc[:5,:32],headers=anonymized_data.columns.to_list(),tablefmt="simple",maxcolwidths=[None, 6]))
print("\n")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")