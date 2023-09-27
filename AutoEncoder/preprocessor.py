import pandas as pd
import joblib
import io
from sklearn.model_selection import train_test_split
from AutoEncoder.bucketfs_client import BucketFS_client

def dataSplitter(input_df,train_ratio,val_ratio,test_ratio,random_seed):
    """
     @brief Split dataset into training, validation and test set
     @param train_ratio: float or int ratio of training data
     @param val_ratio: float or int ratio of validation data
     @param test_ratio: float or int ratio of test data
     @param random_seed: for debugging
    """
    # Calculate the sizes of train, validation, and test sets
    assert(train_ratio+val_ratio+test_ratio) == 1,f"Ratio is not equal to 1, got{train_ratio+val_ratio+test_ratio} instead."
    total_size = len(input_df)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split training, validation, and test sets
    X_train, X_temp = train_test_split(input_df, test_size=(val_size + test_size), random_state=random_seed)    
    X_val, X_test = train_test_split(X_temp, test_size=test_size, random_state=random_seed)
    return X_train,X_val,X_test


def dataPreprocessor(input_df,is_train,continous_columns,categorical_columns,scaler,onehotencoder,save=None,file_path=None):
    """
     @brief apply scaling and encoding to the input dataframe
     @param input_df: Input dataframe    
     @param is_train: if the input data is a train set, use fit_transform(), else use transform()
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param scaler: Scaler object
     @param onehotencoder: Onehot encoder object
     @param save: Enable saving fit-transformed scaler and encoder. Can be "BucketFS" or "local"
     @param file_path: Path of the save destination of scaler/encoder
    """
    if (save is not None):
      assert file_path is not None, "file_path must be declared if save is specified."
    if (file_path is not None):
      assert save is not None, "save must be declared if file_path is specified."

    client = BucketFS_client()
    # Preprocess continous columns
    if (is_train == True):
      input_df_scaled = scaler.fit_transform(input_df[continous_columns])
      if (save is not None):          
        if (save=="BucketFS"):
          # Save to BucketFS 
          buffer = io.BytesIO()
          joblib.dump(scaler,buffer)
          client.upload(f'autoencoder/{file_path}_scaler.pkl', buffer)      
        elif (save=="local"): 
          # Save locally
          joblib.dump(scaler,f'{file_path}_scaler.pkl')
      else:
        pass
    else:
      if (save is not None):
        if (save=="BucketFS"):
          # Load from BucketFS
          data = client.download(f'autoencoder/{file_path}_scaler.pkl')
          scaler = joblib.load(data)
        elif (save=="local"):
          # Load locally
          scaler = joblib.load(f'{file_path}_scaler.pkl')
      else: 
        pass
      input_df_scaled = scaler.transform(input_df[continous_columns])
    input_df[continous_columns] = input_df_scaled

    # Preprocess categorical columns
    if (is_train == True):
      input_df_encoded = onehotencoder.fit_transform(input_df[categorical_columns])
      if (save is not None):  
        if (save=="BucketFS"):
          # Save to BucketFS
          buffer = io.BytesIO()
          joblib.dump(onehotencoder,buffer)
          client.upload(f'autoencoder/{file_path}_encoder.pkl', buffer)     
        elif (save=="local"):
          # Save locally
          joblib.dump(scaler,f'{file_path}_encoder.pkl')
      else:
        pass
    else:
      if (save is not None):
        if (save=="BucketFS"):
          # Load from BucketFS
          data = client.download(f'autoencoder/{file_path}_encoder.pkl')
          scaler = joblib.load(data)
        elif (save=="local"):
          # Load locally
          scaler = joblib.load(f'{file_path}_encoder.pkl')
      else:
        pass
      input_df_encoded = onehotencoder.transform(input_df[categorical_columns])
    input_df_encoded_part = pd.DataFrame(input_df_encoded, columns=onehotencoder.get_feature_names_out(categorical_columns),index=input_df.index)
    input_df = pd.concat([input_df,input_df_encoded_part],axis=1)
    input_df.drop(columns=categorical_columns, inplace=True)

    return input_df