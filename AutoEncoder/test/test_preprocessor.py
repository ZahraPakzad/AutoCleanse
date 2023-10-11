import pandas as pd
import numpy as np
import unittest
import os
from AutoEncoder.preprocessor import *
from AutoEncoder.bucketfs_client import *

class test_preprocessor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_preprocessor, self).__init__(*args, **kwargs)
        self.data = {'Numerical': [11,22,33,44,55,66,77,88,99,00], 'Categorical': ['A','C','B','A','D','C','B','D','D','C']}
        self.df = pd.DataFrame(self.data)

    def test_dataSplitter(self):
        # Test nominal output value
        _,_,test1 = dataSplitter(self.df,0.8,0.1,0.1,42)
        test1_nominal = pd.DataFrame({'Numerical':[22],'Categorical':['C']})
        assert np.array_equal(test1.values,test1_nominal.values), "Unexpected result."

        # Test invalid split ratio
        _,_,test2 = dataSplitter(self.df,0.8,0.2,0.1,42)
        test2_nominal = pd.DataFrame({'Numerical':[99,22],'Categorical':['D','C']})
        assert np.array_equal(test2.values,test2_nominal.values), "Unexpected result."

    def test_dataPreprocessor_BucketFS(self):
        client = BucketFS_client()

        # Test save 
        df_train,_,_ = dataPreprocessor(input_df=self.df,
                                        is_train=True,             
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'],
                                        location="BucketFS",
                                        layers=['test'])                                         

        self.assertTrue(client.check("autoencoder/autoencoder_test_scaler.pkl"))
        self.assertTrue(client.check("autoencoder/autoencoder_test_encoder.pkl"))

        # Test load 
        df_test,_,_ = dataPreprocessor( input_df=self.df,
                                        is_train=False,             
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'],
                                        location="BucketFS",
                                        layers=['test']) 

        if client.check("autoencoder/autoencoder_test_scaler.pkl"):
            client.delete("/autoencoder/autoencoder_test_scaler.pkl")
        if client.check("autoencoder/autoencoder_test_encoder.pkl"):
            client.delete("/autoencoder/autoencoder_test_encoder.pkl")                                                 
      
    def test_dataPreprocessor_local(self):
        # Test save 
        df_train,_,_ = dataPreprocessor(input_df=self.df,
                                        is_train=True,             
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'],
                                        location="local",
                                        layers=['test'])                                         

        self.assertTrue(os.path.exists("autoencoder_test_scaler.pkl"))
        self.assertTrue(os.path.exists("autoencoder_test_encoder.pkl"))

        # Test load 
        df_test,_,_ = dataPreprocessor( input_df=self.df,
                                        is_train=False,             
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'],
                                        location="local",
                                        layers=['test']) 

        if os.path.exists("autoencoder_test_scaler.pkl"):
            os.remove("autoencoder_test_scaler.pkl")
        if os.path.exists("autoencoder_test_encoder.pkl"):
            os.remove("autoencoder_test_encoder.pkl")             

if __name__ == "__main__":
    unittest.main()