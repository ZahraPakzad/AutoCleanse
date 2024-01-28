import pandas as pd
import numpy as np
import unittest
import os
import torchsummary
from AutoCleanse.autoencoder import *
from AutoCleanse.bucketfs_client import *
from AutoCleanse.dataloader import *
from AutoCleanse.preprocessor import *
from sklearn.preprocessing import *
from torch.optim.lr_scheduler import StepLR

class test_autoencoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_autoencoder, self).__init__(*args, **kwargs)    
        df = pd.DataFrame({'Numerical': [11,22,33,44,55,66,77,88,99,00], 'Categorical': ['A','C','B','A','D','C','B','D','D','C']})
        self.preprocessor = Preprocessor(scaler=MinMaxScaler(),encoder=OneHotEncoder(sparse=False))
        train_set,val_set,test_set = self.preprocessor.split(df,0.6,0.2,0.2,42)
        X_train = self.preprocessor.fit_transform(input_df=train_set,
                                             continous_columns=['Numerical'],
                                             categorical_columns=['Categorical'])

        X_val =  self.preprocessor.transform(input_df=val_set,
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'])

        self.X_test = self.preprocessor.transform(input_df=test_set,
                                             continous_columns=['Numerical'],
                                             categorical_columns=['Categorical'])
        train_dataset = PlainDataset(X_train)
        val_dataset = PlainDataset(X_val)
        test_dataset = PlainDataset(self.X_test)   
        def custom_collate_fn(batch):
            tensor_data = torch.stack([item[0] for item in batch])
            indices = [item[1] for item in batch]
            return tensor_data, indices   
        self.categories = self.preprocessor.encoder.categories_
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True,collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)                                    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.autoencoder = Autoencoder(layers=[self.X_test.shape[1],50,20,2],dropout_enc=[(0,0.5)],dropout_dec=[(0,0.5)],batch_norm=False,\
                                       learning_rate=1e-3,weight_decay=1e-5,l1_strength=1e-5,l2_strength=1e-5)                                    

    @pytest.mark.run(order=1)
    def test_train(self):        
        self.autoencoder.train_model(patience=10,
                                    num_epochs=1,
                                    batch_size=2,
                                    categories=self.categories,
                                    train_loader=self.train_loader,
                                    val_loader=self.val_loader,
                                    continous_columns=['Numerical'], 
                                    categorical_columns=['Categorical'], 
                                    device=self.device)

    @pytest.mark.run(order=2)
    def test_clean(self):
        cleaned_data = self.autoencoder.clean(dirty_loader=self.test_loader,
                                              df=self.X_test,
                                              batch_size=1,
                                              continous_columns=['Numerical'],
                                              categorical_columns=['Categorical'],
                                              og_columns=['Numerical','Categorical'],
                                              scaler=self.preprocessor.scaler,
                                              onehotencoder=self.preprocessor.encoder,
                                              device=self.device) 

    @pytest.mark.run(order=3)
    def test_anon(self):
        anonymized_data = self.autoencoder.anonymize(df=self.X_test,
                                                     data_loader=self.test_loader,
                                                     batch_size=1,
                                                     device=self.device)                             

    @pytest.mark.run(order=4)
    def test_save_local(self):        
        self.autoencoder.save("local","test")

    @pytest.mark.run(order=6)
    @pytest.mark.skip
    def test_save_bucketfs(self):
        self.autoencoder.save("bucketfs","test")

    @pytest.mark.run(order=5)
    def test_load_local(self):
        self.autoencoder.load("local","test")

    @pytest.mark.run(order=7)
    @pytest.mark.skip
    def test_load_bucketfs(self):
        self.autoencoder.load("bucketfs","test")

if __name__ == "__main__":
    # unittest.main()
    test = test_autoencoder()
    test.test_train()
    test.test_clean()
    test.test_anon()
    test.test_save_local()
    test.test_load_local()
    test.test_save_bucketfs()
    test.test_load_bucketfs()