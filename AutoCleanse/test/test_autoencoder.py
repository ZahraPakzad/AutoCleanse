import pandas as pd
import numpy as np
import pytest
import os
import torchsummary
from AutoCleanse.autoencoder import *
from AutoCleanse.bucketfs_client import *
from AutoCleanse.dataloader import *
from AutoCleanse.preprocessor import *
from sklearn.preprocessing import *
from torch.optim.lr_scheduler import StepLR

@pytest.fixture
def autoencoder_fixture():
    df = pd.DataFrame({'Numerical': [11,22,33,44,55,66,77,88,99,00], 'Categorical': ['A','C','B','A','D','C','B','D','D','C']})
    preprocessor = Preprocessor(scaler=MinMaxScaler(), encoder=OneHotEncoder(sparse_output=False))
    train_set, val_set, test_set = preprocessor.split(df, 0.6, 0.2, 0.2, 42)
    X_train = preprocessor.fit_transform(input_df=train_set, continous_columns=['Numerical'], categorical_columns=['Categorical'])
    X_val = preprocessor.transform(input_df=val_set, continous_columns=['Numerical'], categorical_columns=['Categorical'])
    X_test = preprocessor.transform(input_df=test_set, continous_columns=['Numerical'], categorical_columns=['Categorical'])
    
    train_dataset = PlainDataset(X_train)
    val_dataset = PlainDataset(X_val)
    test_dataset = PlainDataset(X_test)
    
    def custom_collate_fn(batch):
        tensor_data = torch.stack([item[0] for item in batch])
        indices = [item[1] for item in batch]
        return tensor_data, indices

    categories = preprocessor.encoder.categories_
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder = Autoencoder(layers=[X_test.shape[1], 50, 20, 2], dropout_enc=[(0, 0.5)], dropout_dec=[(0, 0.5)], 
                              batch_norm=False, learning_rate=1e-3, weight_decay=1e-5, l1_strength=1e-5, l2_strength=1e-5)
    
    autoencoder.train_model(patience=10,
                            num_epochs=1,
                            batch_size=2,
                            categories=categories,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            continous_columns=['Numerical'], 
                            categorical_columns=['Categorical'], 
                            device=device)                               

    return {
        'preprocessor': preprocessor,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'X_test': X_test,
        'categories': categories,
        'device': device,
        'autoencoder': autoencoder
    }                                                                                  

@pytest.mark.autoencoder
@pytest.mark.run(order=1)
def test_clean(autoencoder_fixture):
    cleaned_data = autoencoder_fixture['autoencoder'].clean(dirty_loader=autoencoder_fixture['test_loader'],
                                                            df=autoencoder_fixture['X_test'],
                                                            batch_size=1,
                                                            continous_columns=['Numerical'],
                                                            categorical_columns=['Categorical'],
                                                            og_columns=['Numerical','Categorical'],
                                                            scaler=autoencoder_fixture['preprocessor'].scaler,
                                                            onehotencoder=autoencoder_fixture['preprocessor'].encoder,
                                                            device=autoencoder_fixture['device']) 

@pytest.mark.autoencoder
@pytest.mark.run(order=2)
def test_anon(autoencoder_fixture):
    anonymized_data = autoencoder_fixture['autoencoder'].anonymize(df=autoencoder_fixture['X_test'],
                                                                   data_loader=autoencoder_fixture['test_loader'],
                                                                   batch_size=1,
                                                                   device=autoencoder_fixture['device'])                             

@pytest.mark.autoencoder
@pytest.mark.run(order=3)
def test_save_local(autoencoder_fixture):        
    autoencoder_fixture['autoencoder'].save("local","test")

@pytest.mark.autoencoder
@pytest.mark.run(order=4)
def test_load_local(autoencoder_fixture):
    autoencoder_fixture['autoencoder'].load("local","test")

@pytest.mark.autoencoder
@pytest.mark.run(order=5)
@pytest.mark.bucketfs
def test_save_bucketfs(autoencoder_fixture):
    autoencoder_fixture['autoencoder'].save("bucketfs","test")

@pytest.mark.autoencoder
@pytest.mark.run(order=6)
@pytest.mark.bucketfs
def test_load_bucketfs(autoencoder_fixture):
    autoencoder_fixture['autoencoder'].load("bucketfs","test")