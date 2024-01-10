import torch 
import pandas as pd
import torch.nn as nn
import time
import io
import joblib
import argparse

from torchsummary import summary
from tqdm import tqdm
from tabulate import tabulate
from sklearn.preprocessing import *

from AutoCleanse.utils import *
from AutoCleanse.dataloader import PlainDataset, DataLoader
from AutoCleanse.autoencoder import *
from AutoCleanse.loss_model import loss_CEMSE
from AutoCleanse.preprocessor import Preprocessor
from AutoCleanse.anonymize import anonymize
from AutoCleanse.bucketfs_client import *

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

scaler = MinMaxScaler()
onehotencoder = OneHotEncoder(sparse=False)
preprocessor = Preprocessor(scaler,onehotencoder)

X_train,X_val,X_test = preprocessor.split(df=df,
                                        train_ratio=0.7,
                                        val_ratio=0.15,
                                        test_ratio=0.15,
                                        random_seed=42)
X_dirty = replace_with_nan(X_test,0,42)


X_train = preprocessor.fit_transform(input_df=X_train,
                                    continous_columns=continous_columns,
                                    categorical_columns=categorical_columns)

X_val = preprocessor.transform(input_df=X_val,    
                            continous_columns=continous_columns,
                            categorical_columns=categorical_columns)                          

X_test = preprocessor.transform(input_df=X_test,   
                                continous_columns=continous_columns,
                                categorical_columns=categorical_columns)  

X_dirty = preprocessor.transform(input_df=X_dirty,   
                                continous_columns=continous_columns,
                                categorical_columns=categorical_columns) 

# preprocessor.save("test","local")
# preprocessor.save("test","bucketfs")
# preprocessor.load("test","local")
# preprocessor.load("test","bucketfs")

categories = preprocessor.encoder.categories_

# Create dataloader
train_dataset = PlainDataset(X_train)
val_dataset = PlainDataset(X_val)
test_dataset = PlainDataset(X_test)
dity_dataset = PlainDataset(X_dirty)

def custom_collate_fn(batch):
    tensor_data = torch.stack([item[0] for item in batch])
    indices = [item[1] for item in batch]
    return tensor_data, indices

torch.manual_seed(42) #@TODO: random seed 
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(dity_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
test_loader_og = DataLoader(dity_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)

# Declaring model
layers = [X_train.shape[1],1024,128]   
wlc = (1,5)  
# layers = string2list(args.layers) 
# wlc = string2tupple(args.wlc)                           
autoencoder = Autoencoder(layers=layers,dropout_enc=[(0,0.0)],dropout_dec=[(0,0.1)], batch_norm=True, \
                          learning_rate=1e-4,weight_decay=1e-5,l1_strength=1e-5,l2_strength=1e-5)
autoencoder.load("local","test")
summary(autoencoder.to(device),torch.tensor(X_train.values).float().to(device).shape[1:])

# autoencoder.train_model(
#       patience=10,
#       num_epochs=1,
#       batch_size=batch_size,
#       train_loader=train_loader,
#       val_loader=val_loader,
#       continous_columns=continous_columns, 
#       categorical_columns=categorical_columns, 
#       categories=categories,
#       device=device,
#       wlc=wlc)
# autoencoder.save("local","test")

# cleaned_data = autoencoder.clean(test_loader=test_loader,
#                                 test_loader_og=test_loader_og,
#                                 test_df=X_dirty,
#                                 batch_size=batch_size,
#                                 continous_columns=continous_columns, 
#                                 categorical_columns=categorical_columns, 
#                                 og_columns=og_columns,
#                                 onehotencoder=preprocessor.encoder, 
#                                 scaler=preprocessor.scaler,
#                                 device=device)                    

# anonymized_data = autoencoder.anonymize(test_df=X_test,
#                                         test_loader=test_loader,
#                                         batch_size=batch_size,
#                                         device=device) 

# print("\n")
# print(tabulate(df.loc[[28296,28217,8054,4223,22723],og_columns],headers=og_columns,tablefmt="simple",maxcolwidths=[None, 4]))
# print("\n")
# print(tabulate(cleaned_data.loc[[28296,28217,8054,4223,22723]],headers=cleaned_data.columns.to_list(),tablefmt="simple",maxcolwidths=[None, 4]))
# print("\n")
# print(tabulate(anonymized_data.round(decimals=4).iloc[:5,:32],headers=anonymized_data.columns.to_list(),tablefmt="simple",maxcolwidths=[None, 6]))
# print("\n")
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# # cleaned_data.to_csv('df_cleaned.csv',index=False)