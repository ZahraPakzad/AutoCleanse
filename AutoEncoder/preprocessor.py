import pandas as pd
import joblib
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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


def dataPreprocessor(input_df,is_train,continous_columns,categorical_columns,location=None,prefix=None):
    """
     @brief apply scaling and encoding to the input dataframe
     @param input_df: Input dataframe    
     @param is_train: if the input data is a train set, use fit_transform(), else use transform()
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param scaler: Scaler object
     @param onehotencoder: Onehot encoder object
     @param location: Enable saving/loading fit-transformed scaler and encoder to/from the specified location. Can be "BucketFS" or "local"
     @param prefix: Name prefix of saved scaler/encoder
    """
    if (location is not None):
      assert prefix is not None, "prefix must be declared if location is specified."
    if (prefix is not None):
      assert location is not None, "location must be declared if prefix is specified."

    client = BucketFS_client()
    scaler = StandardScaler()
    onehotencoder = OneHotEncoder(sparse=False)
    # Preprocess continous columns
    if (is_train == True):
      input_df_scaled = scaler.fit_transform(input_df[continous_columns])
      if (location is not None):          
        if (location=="BucketFS"):
          # Save to BucketFS 
          buffer = io.BytesIO()
          joblib.dump(scaler,buffer)
          client.upload(f'autoencoder/{prefix}_scaler.pkl', buffer)      
        elif (location=="local"): 
          # Save locally
          joblib.dump(scaler,f'{prefix}_scaler.pkl')
      else:
        pass
    else:
      if (location is not None):
        if (location=="BucketFS"):
          # Load from BucketFS
          data = client.download(f'autoencoder/{prefix}_scaler.pkl')
          try:
            scaler = joblib.load(data)
          except Exception as e:
            print("Fail to load scaler from BucketFS", e)
        elif (location=="local"):
          # Load locally
          try:
            scaler = joblib.load(f'{prefix}_scaler.pkl')
          except Exception as e:
            print("Fail to load encoder from local disk", e)
      else: 
        pass
      input_df_scaled = scaler.transform(input_df[continous_columns])
    input_df[continous_columns] = input_df_scaled

    # Preprocess categorical columns
    if (is_train == True):
      input_df_encoded = onehotencoder.fit_transform(input_df[categorical_columns])
      if (location is not None):  
        if (location=="BucketFS"):
          # Save to BucketFS
          buffer = io.BytesIO()
          joblib.dump(onehotencoder,buffer)
          client.upload(f'autoencoder/{prefix}_encoder.pkl', buffer)     
        elif (location=="local"):
          # Save locally
          joblib.dump(onehotencoder,f'{prefix}_encoder.pkl')
      else:
        pass
    else:
      if (location is not None):
        if (location=="BucketFS"):
          # Load from BucketFS
          data = client.download(f'autoencoder/{prefix}_encoder.pkl')
          try: 
            onehotencoder = joblib.load(data)
          except Exception as e:
            print("Fail to load encoder from BucketFS", e)
        elif (location=="local"):
          # Load locally
          try:
            onehotencoder = joblib.load(f'{prefix}_encoder.pkl')
          except Exception as e:
            print("Fail to load encoder from local disk", e) 
      else:
        pass
      input_df_encoded = onehotencoder.transform(input_df[categorical_columns])
    input_df_encoded_part = pd.DataFrame(input_df_encoded, columns=onehotencoder.get_feature_names_out(categorical_columns),index=input_df.index)
    input_df = pd.concat([input_df,input_df_encoded_part],axis=1)
    input_df.drop(columns=categorical_columns, inplace=True)

    return input_df,scaler,onehotencoder