import pandas as pd
import numpy as np
import os
import pytest
from AutoCleanse.preprocessor import *
from AutoCleanse.bucketfs_client import *
from AutoCleanse.utils import replace_with_nan
from sklearn.preprocessing import *

data = {'Numerical': [11, 22, 33, 44, 55, 66, 77, 88, 99, 00], 
        'Categorical': ['A', 'C', 'B', 'A', 'D', 'C', 'B', 'D', 'D', 'C']}

@pytest.fixture
def preprocessor_fixture():
    return Preprocessor(scaler=MinMaxScaler(), encoder=OneHotEncoder(sparse_output=False))

@pytest.mark.preprocessor
def test_split(preprocessor_fixture):
    df = pd.DataFrame(data)
    _, _, test1 = preprocessor_fixture.split(df, 0.8, 0.1, 0.1, 42)
    test1_nominal = pd.DataFrame({'Numerical': [22], 'Categorical': ['C']})
    assert np.array_equal(test1.values, test1_nominal.values), "Unexpected result."

    _, _, test2 = preprocessor_fixture.split(df, 0.8, 0.2, 0.1, 42)
    test2_nominal = pd.DataFrame({'Numerical': [99, 22], 'Categorical': ['D', 'C']})
    assert np.array_equal(test2.values, test2_nominal.values), "Unexpected result."

    print("test_dataSplitter: OK")

@pytest.mark.preprocessor
@pytest.mark.bucketfs
def test_preprocessor_BucketFS(preprocessor_fixture):
    df = pd.DataFrame(data)

    # Test save
    df_train = preprocessor_fixture.fit_transform(
        input_df=df,
        continous_columns=['Numerical'],
        categorical_columns=['Categorical']
    )
    preprocessor_fixture.save("test1", "bucketfs", url="http://172.18.0.2:6583", bucket="default", user="w", password="write")
    assert BucketFSClient.check(f"preprocessor/preprocessor_test1.pkl")

    # Test load
    preprocessor2 = Preprocessor(scaler=MinMaxScaler(), encoder=OneHotEncoder(sparse_output=False))
    preprocessor2.load("test1", "bucketfs", url="http://172.18.0.2:6583", bucket="default", user="w", password="write")
    df_test = preprocessor2.transform(
        input_df=df,
        continous_columns=['Numerical'],
        categorical_columns=['Categorical']
    )

    if BucketFSClient.check(f"preprocessor/preprocessor_test1.pkl"):
        BucketFSClient.delete(f"/preprocessor/preprocessor_test1.pkl")

    print("test_preprocessor_BucketFS: OK")

@pytest.mark.preprocessor
def test_preprocessor_local(preprocessor_fixture):
    df = pd.DataFrame(data)

    # Test save
    df_train = preprocessor_fixture.fit_transform(
        input_df=df,
        continous_columns=['Numerical'],
        categorical_columns=['Categorical']
    )
    preprocessor_fixture.save("test", "local")
    assert os.path.exists(f"preprocessor_test.pkl")

    # Test load
    preprocessor2 = Preprocessor(scaler=MinMaxScaler(), encoder=OneHotEncoder(sparse_output=False))
    preprocessor2.load("test", "local")
    df_test = preprocessor2.transform(
        input_df=df,
        continous_columns=['Numerical'],
        categorical_columns=['Categorical']
    )

    if os.path.exists(f"preprocessor_test.pkl"):
        os.remove(f"preprocessor_test.pkl")

    print("test_preprocessor_local: OK")

@pytest.mark.preprocessor
@pytest.mark.bucketfs(reason="Skipping test_preprocessor_con_BucketFS")
def test_preprocessor_con_BucketFS(preprocessor_fixture):
    df = pd.DataFrame(data)

    # Test save
    df_train = preprocessor_fixture.fit_transform(
        input_df=df,
        continous_columns=['Numerical']
    )
    preprocessor_fixture.save("test2", "bucketfs", url="http://172.18.0.2:6583", bucket="default", user="w", password="write")
    assert BucketFSClient.check(f"preprocessor/preprocessor_test2.pkl")

    # Test load
    preprocessor2 = Preprocessor(scaler=MinMaxScaler(), encoder=OneHotEncoder(sparse_output=False))
    preprocessor2.load("test2", "bucketfs", url="http://172.18.0.2:6583", bucket="default", user="w", password="write")
    df_test = preprocessor2.transform(
        input_df=df,
        continous_columns=['Numerical']
    )

    if BucketFSClient.check(f"preprocessor/preprocessor_test2.pkl"):
        BucketFSClient.delete(f"preprocessor/preprocessor_test2.pkl")

    print("test_preprocessor_con_BucketFS: OK")

@pytest.mark.preprocessor
def test_preprocessor_con_local(preprocessor_fixture):
    df = pd.DataFrame(data)

    # Test save
    df_train = preprocessor_fixture.fit_transform(
        input_df=df,
        continous_columns=['Numerical']
    )
    preprocessor_fixture.save("test", "local")
    assert os.path.exists(f"preprocessor_test.pkl")

    # Test load
    preprocessor2 = Preprocessor(scaler=MinMaxScaler(), encoder=OneHotEncoder(sparse_output=False))
    preprocessor2.load("test", "local")
    df_test = preprocessor2.transform(
        input_df=df,
        continous_columns=['Numerical']
    )

    if os.path.exists(f"preprocessor_test.pkl"):
        os.remove(f"preprocessor_test.pkl")

    print("test_preprocessor_con_local: OK")

@pytest.mark.preprocessor
@pytest.mark.bucketfs(reason="Skipping test_preprocessor_cat_BucketFS")
def test_preprocessor_cat_BucketFS(preprocessor_fixture):
    df = pd.DataFrame(data)

    # Test save
    df_train = preprocessor_fixture.fit_transform(
        input_df=df,
        categorical_columns=['Categorical']
    )
    preprocessor_fixture.save("test3", "bucketfs", url="http://172.18.0.2:6583", bucket="default", user="w", password="write")
    assert BucketFSClient.check(f"preprocessor/preprocessor_test3.pkl")

    # Test load
    preprocessor2 = Preprocessor(scaler=MinMaxScaler(), encoder=OneHotEncoder(sparse_output=False))
    preprocessor2.load("test3", "bucketfs", url="http://172.18.0.2:6583", bucket="default", user="w", password="write")
    df_test = preprocessor2.transform(
        input_df=df,
        categorical_columns=['Categorical']
    )

    if BucketFSClient.check(f"preprocessor/preprocessor_test3.pkl"):
        BucketFSClient.delete(f"/preprocessor/preprocessor_test3.pkl")

    print("test_preprocessor_cat_BucketFS: OK")

@pytest.mark.preprocessor
def test_preprocessor_cat_local(preprocessor_fixture):
    df = pd.DataFrame(data)

    # Test save
    df_train = preprocessor_fixture.fit_transform(
        input_df=df,
        categorical_columns=['Categorical']
    )
    preprocessor_fixture.save("test", "local")
    assert os.path.exists(f"preprocessor_test.pkl")

    # Test load
    preprocessor2 = Preprocessor(scaler=MinMaxScaler(), encoder=OneHotEncoder(sparse_output=False))
    preprocessor2.load("test", "local")
    df_test = preprocessor2.transform(
        input_df=df,
        categorical_columns=['Categorical']
    )

    if os.path.exists(f"preprocessor_test.pkl"):
        os.remove(f"preprocessor_test.pkl")

    print("test_preprocessor_cat_local: OK")
