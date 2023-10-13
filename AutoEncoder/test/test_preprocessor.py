import pandas as pd
import numpy as np
import unittest
import os
from AutoEncoder.preprocessor import *
from AutoEncoder.bucketfs_client import *

data = {'Numerical': [11,22,33,44,55,66,77,88,99,00], 'Categorical': ['A','C','B','A','D','C','B','D','D','C']}
data_con = {'Numerical': [11,22,33,44,55,66,77,88,99,00]}
data_cat = {'Categorical': ['A','C','B','A','D','C','B','D','D','C']}

class test_preprocessor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_preprocessor, self).__init__(*args, **kwargs)
        self.df = pd.DataFrame(data)
        self.df_con = pd.DataFrame(data_con)
        self.df_cat = pd.DataFrame(data_cat)

    def test_dataSplitter(self):
        # Test nominal output value
        _,_,test1 = dataSplitter(self.df,0.8,0.1,0.1,42)
        test1_nominal = pd.DataFrame({'Numerical':[22],'Categorical':['C']})
        assert np.array_equal(test1.values,test1_nominal.values), "Unexpected result."

        # Test invalid split ratio
        _,_,test2 = dataSplitter(self.df,0.8,0.2,0.1,42)
        test2_nominal = pd.DataFrame({'Numerical':[99,22],'Categorical':['D','C']})
        assert np.array_equal(test2.values,test2_nominal.values), "Unexpected result."

        print("test_dataSplitter: OK")
    
    def test_dataPreprocessor_BucketFS(self):
        client = BucketFS_client()
        location = "BucketFS"
        layers = ['test']

        # Test save 
        df_train,_,_ = dataPreprocessor(input_df=self.df,
                                        is_train=True,             
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'],
                                        location=location,
                                        layers=layers)                                        

        self.assertTrue(client.check(f"autoencoder/autoencoder_{layers[0]}_scaler.pkl"))
        self.assertTrue(client.check(f"autoencoder/autoencoder_{layers[0]}_encoder.pkl"))

        # Test load 
        df_test,_,_ = dataPreprocessor( input_df=self.df,
                                        is_train=False,             
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'],
                                        location=location,
                                        layers=layers)

        if client.check(f"autoencoder/autoencoder_{layers[0]}_scaler.pkl"):
            client.delete(f"/autoencoder/autoencoder_{layers[0]}_scaler.pkl")
        if client.check(f"autoencoder/autoencoder_{layers[0]}_encoder.pkl"):
            client.delete(f"/autoencoder/autoencoder_{layers[0]}_encoder.pkl")      

        print("test_dataPreprocessor_BucketFS: OK")                                           
      
    def test_dataPreprocessor_local(self):
        location = "local"
        layers = ['test']

        # Test save 
        df_train,_,_ = dataPreprocessor(input_df=self.df,
                                        is_train=True,             
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'],
                                        location=location,
                                        layers=layers)                                        

        self.assertTrue(os.path.exists(f"autoencoder_{layers[0]}_scaler.pkl"))
        self.assertTrue(os.path.exists(f"autoencoder_{layers[0]}_encoder.pkl"))

        # Test load 
        df_test,_,_ = dataPreprocessor( input_df=self.df,
                                        is_train=False,             
                                        continous_columns=['Numerical'],
                                        categorical_columns=['Categorical'],
                                        location=location,
                                        layers=layers)

        if os.path.exists(f"autoencoder_{layers[0]}_scaler.pkl"):
            os.remove(f"autoencoder_{layers[0]}_scaler.pkl")
        if os.path.exists(f"autoencoder_{layers[0]}_encoder.pkl"):
            os.remove(f"autoencoder_{layers[0]}_encoder.pkl")  

        print("test_dataPreprocessor_local: OK")

    def test_dataPreprocessor_con_BucketFS(self):
        client = BucketFS_client()
        location="BucketFS"
        layers=['test_con']

        # Test save 
        df_train,_,_ = dataPreprocessor(input_df=self.df,
                                        is_train=True,             
                                        continous_columns=['Numerical'],
                                        location=location,
                                        layers=layers)                                         

        self.assertTrue(client.check(f"autoencoder/autoencoder_{layers[0]}_scaler.pkl"))

        # Test load 
        df_test,_,_ = dataPreprocessor( input_df=self.df,
                                        is_train=False,             
                                        continous_columns=['Numerical'],
                                        location=location,
                                        layers=layers)

        if client.check(f"autoencoder/autoencoder_{layers[0]}_scaler.pkl"):
            client.delete(f"/autoencoder/autoencoder_{layers[0]}_scaler.pkl")   

        print("test_dataPreprocessor_con_BucketFS: OK")  

    def test_dataPreprocessor_con_local(self):
        location="local"
        layers=['test_con']

        # Test save 
        df_train,_,_ = dataPreprocessor(input_df=self.df,
                                        is_train=True,             
                                        continous_columns=['Numerical'],                                        
                                        location=location,
                                        layers=layers)                                         

        self.assertTrue(os.path.exists(f"autoencoder_{layers[0]}_scaler.pkl"))

        # Test load 
        df_test,_,_ = dataPreprocessor( input_df=self.df,
                                        is_train=False,             
                                        continous_columns=['Numerical'],
                                        location=location,
                                        layers=layers)

        if os.path.exists(f"autoencoder_{layers[0]}_scaler.pkl"):
            os.remove(f"autoencoder_{layers[0]}_scaler.pkl")

        print("test_dataPreprocessor_con_local: OK")

    def test_dataPreprocessor_cat_BucketFS(self):
        client = BucketFS_client()
        location="BucketFS"
        layers=['test_cat']

        # Test save 
        df_train,_,_ = dataPreprocessor(input_df=self.df,
                                        is_train=True,             
                                        categorical_columns=['Categorical'],
                                        location=location,
                                        layers=layers)                                         

        self.assertTrue(client.check(f"autoencoder/autoencoder_{layers[0]}_encoder.pkl"))

        # Test load 
        df_test,_,_ = dataPreprocessor( input_df=self.df,
                                        is_train=False,             
                                        categorical_columns=['Categorical'],
                                        location=location,
                                        layers=layers)

        if client.check(f"autoencoder/autoencoder_{layers[0]}_encoder.pkl"):
            client.delete(f"/autoencoder/autoencoder_{layers[0]}_encoder.pkl")      

        print("test_dataPreprocessor_cat_BucketFS: OK")  

    def test_dataPreprocessor_cat_local(self):
        location = "local"
        layers = ['test_cat']

        # Test save 
        df_train,_,_ = dataPreprocessor(input_df=self.df,
                                        is_train=True,             
                                        categorical_columns=['Categorical'],
                                        location=location,
                                        layers=layers)                                        

        self.assertTrue(os.path.exists(f"autoencoder_{layers[0]}_encoder.pkl"))

        # Test load 
        df_test,_,_ = dataPreprocessor( input_df=self.df,
                                        is_train=False,             
                                        categorical_columns=['Categorical'],
                                        location=location,
                                        layers=layers)

        if os.path.exists(f"autoencoder_{layers[0]}_encoder.pkl"):
            os.remove(f"autoencoder_{layers[0]}_encoder.pkl")  

        print("test_dataPreprocessor_cat_local: OK")

if __name__ == "__main__":
    unittest.main()