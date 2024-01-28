import pandas as pd
import numpy as np
import unittest
import os
from AutoCleanse.preprocessor import *
from AutoCleanse.bucketfs_client import *
from AutoCleanse.utils import replace_with_nan
from sklearn.preprocessing import *

data = {'Numerical': [11,22,33,44,55,66,77,88,99,00], 
        'Categorical': ['A','C','B','A','D','C','B','D','D','C']}

class test_preprocessor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_preprocessor, self).__init__(*args, **kwargs)
        self.df = pd.DataFrame(data)
        self.preprocessor = Preprocessor(scaler=MinMaxScaler(),encoder=OneHotEncoder(sparse=False))

    @pytest.mark.run(order=1)
    def test_split(self):
        # Test nominal output value
        _,_,test1 = self.preprocessor.split(self.df,0.8,0.1,0.1,42)
        test1_nominal = pd.DataFrame({'Numerical':[22],'Categorical':['C']})
        assert np.array_equal(test1.values,test1_nominal.values), "Unexpected result."

        # Test invalid split ratio
        _,_,test2 = self.preprocessor.split(self.df,0.8,0.2,0.1,42)
        test2_nominal = pd.DataFrame({'Numerical':[99,22],'Categorical':['D','C']})
        assert np.array_equal(test2.values,test2_nominal.values), "Unexpected result."

        print("test_dataSplitter: OK")
    
    @pytest.mark.run(order=2)
    @pytest.mark.skip
    def test_preprocessor_BucketFS(self):

        # Test save 
        df_train = self.preprocessor.fit_transform(input_df=self.df,
                                                   continous_columns=['Numerical'],
                                                   categorical_columns=['Categorical'])                                   
        self.preprocessor.save("test1","bucketfs")
        self.assertTrue(bucketfs_client.check(f"preprocessor/preprocessor_test1.pkl"))

        # Test load 
        preprocessor2 = Preprocessor(scaler=MinMaxScaler(),encoder=OneHotEncoder(sparse=False))
        preprocessor2.load("test1","bucketfs")
        df_test = preprocessor2.transform(input_df=self.df,
                                          continous_columns=['Numerical'],
                                          categorical_columns=['Categorical'])

        if bucketfs_client.check(f"preprocessor/preprocessor_test1.pkl"):
            bucketfs_client.delete(f"/preprocessor/preprocessor_test1.pkl")     

        print("test_preprocessor_BucketFS: OK")                                           
      
    @pytest.mark.run(order=3)
    def test_preprocessor_local(self):

        # Test save 
        df_train = self.preprocessor.fit_transform(input_df=self.df,
                                                   continous_columns=['Numerical'],
                                                   categorical_columns=['Categorical'])                                        
        self.preprocessor.save("test","local")
        self.assertTrue(os.path.exists(f"preprocessor_test.pkl"))

        # Test load 
        preprocessor2 = Preprocessor(scaler=MinMaxScaler(),encoder=OneHotEncoder(sparse=False))
        preprocessor2.load("test","local")
        df_test = preprocessor2.transform(input_df=self.df,
                                          continous_columns=['Numerical'],
                                          categorical_columns=['Categorical'])

        if os.path.exists(f"preprocessor_test.pkl"):
            os.remove(f"preprocessor_test.pkl")

        print("test_preprocessor_local: OK")

    @pytest.mark.run(order=4)
    @pytest.mark.skip
    def test_preprocessor_con_BucketFS(self):

        # Test save 
        df_train = self.preprocessor.fit_transform(input_df=self.df,
                                                   continous_columns=['Numerical'])
        self.preprocessor.save("test2","bucketfs")
        self.assertTrue(bucketfs_client.check(f"preprocessor/preprocessor_test2.pkl"))

        # Test load 
        preprocessor2 = Preprocessor(scaler=MinMaxScaler(),encoder=OneHotEncoder(sparse=False))
        preprocessor2.load("test2","bucketfs")
        df_test = preprocessor2.transform(input_df=self.df,                                              
                                          continous_columns=['Numerical'])

        if bucketfs_client.check(f"preprocessor/preprocessor_test2.pkl"):
            bucketfs_client.delete(f"preprocessor/preprocessor_test2.pkl")   

        print("test_preprocessor_con_BucketFS: OK")  

    @pytest.mark.run(order=5)
    def test_preprocessor_con_local(self):

        # Test save 
        df_train = self.preprocessor.fit_transform(input_df=self.df,
                                                   continous_columns=['Numerical'])                                       
        self.preprocessor.save("test","local")
        self.assertTrue(os.path.exists(f"preprocessor_test.pkl"))

        # Test load 
        preprocessor2 = Preprocessor(scaler=MinMaxScaler(),encoder=OneHotEncoder(sparse=False))
        preprocessor2.load("test","local")
        df_test = preprocessor2.transform(input_df=self.df,
                                          continous_columns=['Numerical'])

        if os.path.exists(f"preprocessor_test.pkl"):
            os.remove(f"preprocessor_test.pkl")

        print("test_preprocessor_con_local: OK")

    @pytest.mark.run(order=6)
    @pytest.mark.skip
    def test_preprocessor_cat_BucketFS(self):

        # Test save 
        df_train = self.preprocessor.fit_transform(input_df=self.df,
                                                   categorical_columns=['Categorical'])                                       
        self.preprocessor.save("test3","bucketfs")
        self.assertTrue(bucketfs_client.check(f"preprocessor/preprocessor_test3.pkl"))

        # Test load
        preprocessor2 = Preprocessor(scaler=MinMaxScaler(),encoder=OneHotEncoder(sparse=False))
        preprocessor2.load("test3","bucketfs")
        df_test = preprocessor2.transform(input_df=self.df,
                                          categorical_columns=['Categorical'])

        if bucketfs_client.check(f"preprocessor/preprocessor_test3.pkl"):
            bucketfs_client.delete(f"/preprocessor/preprocessor_test3.pkl")      

        print("test_preprocessor_cat_BucketFS: OK")  

    @pytest.mark.run(order=7)
    def test_preprocessor_cat_local(self):

        # Test save 
        df_train = self.preprocessor.fit_transform(input_df=self.df,
                                                   categorical_columns=['Categorical'])                                       
        self.preprocessor.save("test","local")
        self.assertTrue(os.path.exists(f"preprocessor_test.pkl"))

        # Test load 
        preprocessor2 = Preprocessor(scaler=MinMaxScaler(),encoder=OneHotEncoder(sparse=False))
        preprocessor2.load("test","local")
        df_test = preprocessor2.transform(input_df=self.df,
                                          categorical_columns=['Categorical'])

        if os.path.exists(f"preprocessor_test.pkl"):
            os.remove(f"preprocessor_test.pkl")  

        print("test_preprocessor_cat_local: OK")

if __name__ == "__main__":
    # unittest.main()
    test = test_preprocessor()
    test.test_preprocessor_local()
    test.test_preprocessor_con_local()
    test.test_preprocessor_cat_local()
    test.test_preprocessor_BucketFS()
    test.test_preprocessor_con_BucketFS()
    test.test_preprocessor_cat_BucketFS()