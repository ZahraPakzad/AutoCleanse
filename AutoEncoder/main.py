import torch 
import pandas as pd
import torch.nn as nn
from utils.utils import argmax, softmax
from data.dataloader import MyDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from tqdm import tqdm
from AutoEncoder.model.autoencoder import build_autoencoder
from AutoEncoder.model.loss_model import loss_CEMSE
from AutoEncoder.data.preprocessor import dataPreprocessor
from AutoEncoder.train import train
from AutoEncoder.clean import clean
from AutoEncoder.anonymize import anonymize
import time

start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading dataset
df = pd.read_csv('~/dataset/categorical/adult.csv')

continous_columns = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']
categorical_columns = ['workclass','education','marital.status','occupation','relationship','race','sex','native.country','income']
df = df[continous_columns+categorical_columns]

scaler = StandardScaler()
onehotencoder = OneHotEncoder(sparse=False)

X_train,X_val,X_test = dataPreprocessor(input_df=df,
                                        train_ratio=0.7,
                                        val_ratio=0.15,
                                        test_ratio=0.15,
                                        continous_columns=continous_columns,
                                        categorical_columns=categorical_columns,
                                        scaler=scaler,
                                        onehotencoder=onehotencoder)

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
autoencoder, encoder, decoder, optimizer = build_autoencoder(layers,dropout=[(0,0.1)])
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)   # LR*0.1 every 30 epochs

autoencoder.to(device)
summary(autoencoder, (X_train.shape))

train(autoencoder=autoencoder,
      patience=15,
      num_epochs=1,
      batch_size=batch_size,
      layers=layers,
      train_loader=train_loader,
      val_loader=val_loader,
      continous_columns=continous_columns, 
      categorical_columns=categorical_columns, 
      onehotencoder=onehotencoder, 
      scaler=scaler,
      optimizer=optimizer,
      scheduler=scheduler,
      device=device)

cleaned_data = clean(autoencoder=autoencoder,
                     test_loader=test_loader,
                     test_df=X_test,
                     batch_size=batch_size,
                     continous_columns=continous_columns, 
                     categorical_columns=categorical_columns, 
                     onehotencoder=onehotencoder, 
                     device=device)

anonymized_data = anonymize(encoder=encoder,
                            test_df=X_test,
                            test_loader=test_loader,
                            batch_size=batch_size,
                            device=device)                     

print(cleaned_data.round(decimals=5).head())
print(anonymized_data.round(decimals=5).head())

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")