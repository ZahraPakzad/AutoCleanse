import pandas as pd
import joblib
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from AutoEncoder.bucketfs_client import BucketFS_client
from AutoEncoder.utils import generate_suffix

def dataSplitter(input_df,train_ratio: float,val_ratio: float,test_ratio: float,random_seed: float):
    """
     @brief Split dataset into training, validation and test set
     @param train_ratio: float or int ratio of training data
     @param val_ratio: float or int ratio of validation data
     @param test_ratio: float or int ratio of test data
     @param random_seed: for debugging
    """
    # Calculate the sizes of train, validation, and test sets
    try: 
      sum = train_ratio + val_ratio + test_ratio
      if (sum != 1):
        raise ValueError(f"Total percentage is not equal to 1, got {sum}")
    except ValueError as e:
      # print(f"{e}. Using default split ratio 0.7:0.15:0.15 instead.")
      train_ratio,val_ratio,test_ratio = 0.7,0.15,0.15
    total_size = len(input_df)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split training, validation, and test sets
    X_train, X_temp = train_test_split(input_df, test_size=(val_size + test_size), random_state=random_seed)    
    X_val, X_test = train_test_split(X_temp, test_size=test_size, random_state=random_seed)
    return X_train,X_val,X_test


def dataPreprocessor(input_df,is_train: bool,layers: list,continous_columns: list,categorical_columns: list,load_method: str="local"):
    """
     @brief apply scaling and encoding to the input dataframe
     @param input_df: Input dataframe    
     @param is_train: if the input data is a train set, use fit_transform(), else use transform()
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param scaler: Scaler object
     @param onehotencoder: Onehot encoder object
     @param load_method: Enable saving/loading fit-transformed scaler and encoder to/from the specified location. Can be "BucketFS" or "local". 
    """
    if (load_method not in {"BucketFS","local"}):
      raise ValueError("location must be either BucketFS or local")

    client = None
    if (load_method == "BucketFS"):
      client = BucketFS_client()

    scaler = MinMaxScaler()
    onehotencoder = OneHotEncoder(sparse=False)

    layers=[0]+layers #@TODO: just an ugly hack using padding to get the correct name for saved weight.
    prefix = generate_suffix(layers,'autoencoder',load_method)[:-4] # Name prefix of saved scaler/encoder
    # Preprocess continous columns
    if (len(continous_columns)!=0):
      if (is_train == True):
        input_df_scaled = scaler.fit_transform(input_df[continous_columns])
        if (load_method=="BucketFS"):
          # Save to BucketFS 
          buffer = io.BytesIO()
          joblib.dump(scaler,buffer)
          try:
            client.upload(f'{prefix}_scaler.pkl', buffer)      
          except Exception as e:
            raise RuntimeError(f"Failed uploading {prefix}_scaler.pkl to BucketFS") from e
        elif (load_method=="local"): 
          # Save locally
          try:
            joblib.dump(scaler,f'{prefix}_scaler.pkl')
          except Exception as e:
            raise RuntimeError(f"Failed saving {prefix}_scaler.pkl to local") from e
      else:
        if (load_method=="BucketFS"):
          # Load from BucketFS
          data = client.download(f'{prefix}_scaler.pkl')
          try:
            scaler = joblib.load(data)
          except Exception as e:
            raise RuntimeError(f"Failed loading {prefix}_scaler.pkl  from BucketFS") from e
        elif (load_method=="local"):
          # Load locally
          try:
            scaler = joblib.load(f'{prefix}_scaler.pkl')
          except Exception as e:
            raise RuntimeError(f"Failed loading {prefix}_scaler.pkl  from BucketFS") from e
        input_df_scaled = scaler.transform(input_df[continous_columns])
      input_df[continous_columns] = input_df_scaled

    # Preprocess categorical columns
    if (len(categorical_columns)!=0):
      if (is_train == True):
        input_df_encoded = onehotencoder.fit_transform(input_df[categorical_columns])
        if (load_method=="BucketFS"):
          # Save to BucketFS
          buffer = io.BytesIO()
          joblib.dump(onehotencoder,buffer)
          try:
            client.upload(f'{prefix}_encoder.pkl', buffer)     
          except Exception as e:
            raise RuntimeError(f"Failed uploading {prefix}_encoder.pkl to BucketFS") from e
        elif (load_method=="local"):
          # Save locally
          try:
            joblib.dump(onehotencoder,f'{prefix}_encoder.pkl')
          except Exception as e:
            raise RuntimeError(f"Failed saving {prefix}_encoder.pkl to local") from e
      else:
        if (load_method=="BucketFS"):
          # Load from BucketFS
          data = client.download(f'{prefix}_encoder.pkl')
          try: 
            onehotencoder = joblib.load(data)
          except Exception as e:
            raise RuntimeError(f"Failed loading {prefix}_encoder.pkl  from BucketFS") from e
        elif (load_method=="local"):
          # Load locally
          try:
            onehotencoder = joblib.load(f'{prefix}_encoder.pkl')
          except Exception as e:
            raise RuntimeError(f"Failed loading {prefix}_encoder.pkl  from BucketFS") from e
        input_df_encoded = onehotencoder.transform(input_df[categorical_columns])
      input_df_encoded_part = pd.DataFrame(input_df_encoded, columns=onehotencoder.get_feature_names_out(categorical_columns),index=input_df.index)
      input_df = pd.concat([input_df,input_df_encoded_part],axis=1)
      input_df.drop(columns=categorical_columns, inplace=True)

    nan_columns = [col for col in input_df.columns if '_nan' in col]
    input_df.drop(columns=nan_columns, inplace=True)
    input_df.fillna(0.0, inplace=True)

    return input_df,scaler,onehotencoder