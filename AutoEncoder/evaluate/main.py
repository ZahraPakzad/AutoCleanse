import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from torchsummary import summary

from AutoEncoder.preprocessor import *
from AutoEncoder.utils import *
from AutoEncoder.dataloader import ClfDataset, DataLoader
from AutoEncoder.evaluate.classifier import *
from AutoEncoder.preprocessor import dataSplitter,dataPreprocessor

from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline
from collections import Counter

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# df = pd.read_csv('~/dataset/categorical/adult.csv').drop(columns=['fnlwgt','capital.gain','capital.loss'])
df = pd.read_csv('~/dataset/categorical/adult.csv').drop(columns=['fnlwgt'])
og_columns = df.columns.to_list()

continous_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist() 
categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
target_columns = ['income']

X = df[continous_columns+categorical_columns]
y = df[target_columns]
# model = ClassifierDummy()
# model.compute_scores(X,y)

X_train,X_val,X_test = dataSplitter(input_df=X,
                                    train_ratio=0.7,
                                    val_ratio=0.15,
                                    test_ratio=0.15,
                                    random_seed=42)
X_dirty = replace_with_nan(X_test,0,42)

y_train,y_val,y_test = dataSplitter(input_df=y,
                                    train_ratio=0.7,
                                    val_ratio=0.15,
                                    test_ratio=0.15,
                                    random_seed=42)
y_dirty = replace_with_nan(y_test,0,42)

layers = [400,200]

X_train,scaler,onehotencoder = dataPreprocessor(
                        input_df=X_train,
                        is_train=True,             
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        load_method="local",
                        layers=layers)                     

X_val,_,_ = dataPreprocessor(
                        input_df=X_val,    
                        is_train=False,         
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        load_method="local",
                        layers=layers)                               

X_test,_,_ = dataPreprocessor(
                        input_df=X_test,   
                        is_train=False,
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        load_method="local",
                        layers=layers)     

X_dirty,_,_ = dataPreprocessor(
                        input_df=X_dirty,   
                        is_train=False,
                        continous_columns=continous_columns,
                        categorical_columns=categorical_columns,
                        load_method="local",
                        layers=layers)  

y_encoder = OneHotEncoder(sparse=False)
y_train = pd.DataFrame(y_encoder.fit_transform(y_train),columns=y_encoder.get_feature_names_out(target_columns),index=y_train.index)
y_val = pd.DataFrame(y_encoder.transform(y_val),columns=y_encoder.get_feature_names_out(target_columns),index=y_val.index)
y_test = pd.DataFrame(y_encoder.transform(y_test),columns=y_encoder.get_feature_names_out(target_columns),index=y_test.index)
y_dirty = pd.DataFrame(y_encoder.transform(y_dirty),columns=y_encoder.get_feature_names_out(target_columns),index=y_dirty.index)

# # Create dataloader
train_dataset = ClfDataset(X_train,y_train)
val_dataset = ClfDataset(X_val,y_val)
test_dataset = ClfDataset(X_test,y_test)
dirty_dataset = ClfDataset(X_dirty,y_dirty)

def custom_collate_fn(batch):
    tensor_data = torch.stack([item[0] for item in batch])
    # Check if tensor_targets are scalars or tensors
    if torch.is_tensor(batch[0][1]):  
        tensor_targets = torch.stack([item[1] for item in batch])
    else:
        tensor_targets = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    
    indices = [item[2] for item in batch]
    return tensor_data, tensor_targets, indices

batch_size = 1000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
dirty_loader = DataLoader(dirty_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)

layers = [X_train.shape[1]]+layers
        
model = ClsNNBase(layers=layers,dropout=[(0,0.0)], batch_norm=False, device=device, \
                  learning_rate=1e-3,weight_decay=1e-5,l1_strength=1e-5,l2_strength=1e-5, \
                  load_method=None)
summary(model.to(device),torch.tensor(X_train.values).float().to(device).shape[1:])

model.train(patience=5,
            num_epochs=5,
            batch_size=batch_size,
            layers=layers,
            train_loader=train_loader,
            val_loader=val_loader,
            continous_columns=continous_columns, 
            categorical_columns=categorical_columns[:-1], # exclude target columns
            onehotencoder=onehotencoder, 
            scaler=scaler,
            device=device,
            save="local")


