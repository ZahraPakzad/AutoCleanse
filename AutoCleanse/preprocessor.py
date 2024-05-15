import pandas as pd
import torch 
import io
import joblib
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.base import clone
# from AutoCleanse.bucketfs_client import bucketfs_client
# from AutoCleanse.utils import *

from bucketfs_client import BucketFSClient
from utils import *


class Preprocessor():
  def __init__(self, scaler, encoder, continous_columns, categorical_columns):
    self.scaler = clone(scaler)
    self.encoder = clone(encoder)
    self.continous_columns = continous_columns
    self.categorical_columns = categorical_columns

  def split(self,df,train_ratio: float,val_ratio: float,test_ratio: float,random_seed: float):
      # Calculate the sizes of train, validation, and test sets
      try: 
        sum = train_ratio + val_ratio + test_ratio
        if (sum != 1):
          raise ValueError(f"Total percentage is not equal to 1, got {sum}")
      except ValueError as e:
        print(f"{e}. Using default split ratio 0.7:0.15:0.15 instead.")
        train_ratio,val_ratio,test_ratio = 0.7,0.15,0.15
      total_size = len(df)
      train_size = int(total_size * train_ratio)
      val_size = int(total_size * val_ratio)
      test_size = total_size - train_size - val_size

      # Split training, validation, and test sets
      df_train, temp = train_test_split(df, test_size=(val_size + test_size), random_state=random_seed)    
      df_val, df_test = train_test_split(temp, test_size=test_size, random_state=random_seed)
      return df_train,df_val,df_test

  def fit_transform(self,input_df):
    # Preprocess continous columns
    if (self.continous_columns is not None):      
      input_df_scaled = self.scaler.fit_transform(input_df[self.continous_columns])
      input_df[self.continous_columns] = input_df_scaled
        
    # Preprocess categorical columns
    if (self.categorical_columns is not None):
      input_df_encoded = self.encoder.fit_transform(input_df[self.categorical_columns])  
      input_df_encoded_part = pd.DataFrame(input_df_encoded, columns=self.encoder.get_feature_names_out(self.categorical_columns),index=input_df.index)
      input_df = pd.concat([input_df,input_df_encoded_part],axis=1)
      input_df.drop(columns=self.categorical_columns, inplace=True)

    nan_columns = [col for col in input_df.columns if '_nan' in col]
    input_df.drop(columns=nan_columns, inplace=True)
    input_df.fillna(0.0, inplace=True)

    return input_df
  
  def transform(self,input_df,method = None, MAE_pack=None):

    # transforming the categorical columns first as when applying the MAE option, the number of columns need to be the the same as
    # the number of encoded categorical columns + the number of continuous columns, in the forward function of MAE
    # Preprocess categorical columns
    if (self.categorical_columns is not None):
      input_df_encoded = self.encoder.transform(input_df[self.categorical_columns])
      input_df_encoded_part = pd.DataFrame(input_df_encoded, columns=self.encoder.get_feature_names_out(self.categorical_columns),index=input_df.index)
      input_df = pd.concat([input_df,input_df_encoded_part],axis=1)
      input_df.drop(columns=self.categorical_columns, inplace=True)

      # Handle NaN in categorical columns
      nan_columns = [col for col in self.categorical_columns if '_nan' in col]
      input_df.drop(columns=nan_columns, inplace=True)


    # Preprocess continous columns
    if (self.continous_columns is not None):

      input_df_scaled = self.scaler.transform(input_df[self.continous_columns])
      input_df[self.continous_columns] = input_df_scaled

      # Handle NaN in continous columns
      if method is not None:

        if MAE_pack is not None and method=="MAE": # passed in for the dirty dataset to replace the NAN values with generated values from MAE
          og_columns = input_df[self.continous_columns].columns.to_list() # todo : just pass the values for the continuous, what is it used for? 
          #doesn't matter, I could just send in nothing as it will not reach a part of code to be useful

          input_df[self.continous_columns] = replace_nan_values(input_df[self.continous_columns], continous_columns=self.continous_columns, og_columns=og_columns, method=method, MAE_pack=MAE_pack, encoder=None, scaler=self.scaler) 

        else: 
          for col in self.continous_columns:
            input_df[col] = replace_nan_values(input_df, col, continous_columns=self.continous_columns, categorical_columns=self.categorical_columns, method=method)    

    return input_df

  def save(self,name,location):
    if (location=="local"):
      try:
        joblib.dump(self,f'preprocessor_{name}.pkl')
      except Exception as e:
        raise RuntimeError(f"Failed saving preprocessor_{name}.pkl to local") from e
    elif (location=="bucketfs"):
      buffer = io.BytesIO()
      joblib.dump(self,buffer)
      max_retries = 3
      delay = 3
      for attempt in range(max_retries):
        try:
          bucketfs_client().upload(f'preprocessor/preprocessor_{name}.pkl',buffer)
          break
        except Exception as e:
          print(f"Upload attempt {attempt + 1} failed. Retrying...\n")
          if (attempt == max_retries - 1):
            raise RuntimeError(f"Failed saving preprocessor_{name}.pkl to BucketFS") from e
          time.sleep(delay)

  def load(self,name,location):
    if (location=="local"):
      try:
        loaded_preprocessor = joblib.load(f'preprocessor_{name}.pkl')
        self.scaler = loaded_preprocessor.scaler
        self.encoder = loaded_preprocessor.encoder
      except Exception as e:
        raise RuntimeError(f"Failed loading preprocessor_{name}.pkl from local") from e
    elif (location=="bucketfs"):
      data = bucketfs_client().download(f'preprocessor/preprocessor_{name}.pkl')
      try:
        loaded_preprocessor = joblib.load(data)
        self.scaler = loaded_preprocessor.scaler
        self.encoder = loaded_preprocessor.encoder
      except Exception as e:
        raise RuntimeError(f"Failed loading preprocessor_{name}.pkl from BucketFS") from e
  
