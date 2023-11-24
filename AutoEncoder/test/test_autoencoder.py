import pandas as pd
import numpy as np
import unittest
import os
import torchsummary
from AutoEncoder.autoencoder import *
from AutoEncoder.train import *
from AutoEncoder.clean import *
from AutoEncoder.anonymize import *
from AutoEncoder.bucketfs_client import *
from AutoEncoder.dataloader import *
from AutoEncoder.preprocessor import *
from torch.optim.lr_scheduler import StepLR

class test_autoencoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_autoencoder, self).__init__(*args, **kwargs)    
        df = pd.DataFrame({'Numerical': [11,22,33,44,55,66,77,88,99,00], 'Categorical': ['A','C','B','A','D','C','B','D','D','C']})
        train_set,val_set,test_set = dataSplitter(df,0.8,0.1,0.1,42)
        layers = [50,20,2]
        X_train,self.scaler,self.onehotencoder = dataPreprocessor(  input_df=train_set,
                                                                    is_train=True,
                                                                    continous_columns=['Numerical'],
                                                                    categorical_columns=['Categorical'],
                                                                    location='local',
                                                                    layers=layers)                                                                  
        X_val,_,_ = dataPreprocessor(   input_df=val_set,
                                        is_train=False,
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'],
                                        location='local',
                                        layers=layers)
        self.X_test,_,_ = dataPreprocessor( input_df=test_set,
                                            is_train=False,
                                            continous_columns=['Numerical'],
                                            categorical_columns=['Categorical'],
                                            location='local',
                                            layers=layers)
        if os.path.exists(f"autoencoder_50_20_2_scaler.pkl"):
            os.remove(f"autoencoder_50_20_2_scaler.pkl")
        if os.path.exists(f"autoencoder_50_20_2_encoder.pkl"):
            os.remove(f"autoencoder_50_20_2_encoder.pkl")    
        train_dataset = TargetlessDataset(X_train)
        val_dataset = TargetlessDataset(X_val)
        test_dataset = TargetlessDataset(self.X_test)     
        def custom_collate_fn(batch):
            tensor_data = torch.stack([item[0] for item in batch])
            indices = [item[1] for item in batch]
            return tensor_data, indices   
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True,collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)                                    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = [self.X_test.shape[1]]+layers
        self.dropout = [(0,0.5)]

    def test_build(self):
        autoencoder, encoder, decoder, optimizer = build_autoencoder(self.layers,self.dropout,learning_rate=1e-3,weight_decay=1e-5,load_method=None)

    def test_build_local(self):
        autoencoder, encoder, decoder, optimizer = build_autoencoder(self.layers,self.dropout,learning_rate=1e-3,weight_decay=1e-5,load_method="local")

    def test_build_bucketfs(self):
        autoencoder, encoder, decoder, optimizer = build_autoencoder(self.layers,self.dropout,learning_rate=1e-3,weight_decay=1e-5,load_method="BucketFS") 

    def test_train(self):        
        autoencoder, encoder, decoder, optimizer = build_autoencoder(self.layers,self.dropout,learning_rate=1e-3,weight_decay=1e-5,load_method=None)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        autoencoder.to(self.device)
        train(  autoencoder=autoencoder,
                patience=15,
                num_epochs=1,
                batch_size=1,
                layers=self.layers,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                continous_columns=['Numerical'], 
                categorical_columns=['Categorical'], 
                onehotencoder=self.onehotencoder, 
                scaler=self.scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                save=None)
    
    def test_train_local(self):
        autoencoder, encoder, decoder, optimizer = build_autoencoder(self.layers,self.dropout,learning_rate=1e-3,weight_decay=1e-5,load_method=None)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        autoencoder.to(self.device)
        train(  autoencoder=autoencoder,
                patience=15,
                num_epochs=1,
                batch_size=1,
                layers=self.layers,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                continous_columns=['Numerical'], 
                categorical_columns=['Categorical'], 
                onehotencoder=self.onehotencoder, 
                scaler=self.scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                save="local")

    def test_train_bucketfs(self):
        autoencoder, encoder, decoder, optimizer = build_autoencoder(self.layers,self.dropout,learning_rate=1e-3,weight_decay=1e-5,load_method=None)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        autoencoder.to(self.device)
        train(  autoencoder=autoencoder,
                patience=15,
                num_epochs=1,
                batch_size=1,
                layers=self.layers,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                continous_columns=['Numerical'], 
                categorical_columns=['Categorical'], 
                onehotencoder=self.onehotencoder, 
                scaler=self.scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                save="BucketFS")

    def test_clean(self):
        autoencoder, encoder, decoder, optimizer = build_autoencoder(self.layers,self.dropout,learning_rate=1e-3,weight_decay=1e-5,load_method=None)
        autoencoder.to(self.device)
        cleaned_data = clean(autoencoder=autoencoder,
                            test_loader=self.test_loader,
                            test_df=self.X_test,
                            batch_size=1,
                            continous_columns=['Numerical'], 
                            categorical_columns=['Categorical'], 
                            og_columns=['Numerical','Categorical'],
                            onehotencoder=self.onehotencoder, 
                            scaler=self.scaler,
                            device=self.device) 

    def test_anon(self):
        autoencoder, encoder, decoder, optimizer = build_autoencoder(self.layers,self.dropout,learning_rate=1e-3,weight_decay=1e-5,load_method=None)
        encoder.to(self.device)
        anonymized_data = anonymize(encoder=encoder,
                                    test_df=self.X_test,
                                    test_loader=self.test_loader,
                                    batch_size=1,
                                    device=self.device)                             

if __name__ == "__main__":
    unittest.main()