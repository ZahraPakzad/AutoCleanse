import pandas as pd
import io
import joblib
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from AutoCleanse.bucketfs_client import bucketfs_client
from AutoCleanse.utils import *
from ordered_set import OrderedSet

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

  def transform(self,input_df):
    # Preprocess continous columns
    if (self.continous_columns is not None):      
      input_df_scaled = self.scaler.transform(input_df[self.continous_columns])
      input_df[self.continous_columns] = input_df_scaled

      # Handle NaN in continous columns
      # input_df.fillna(generate_random_spike(1000, 10000), inplace=True)
      
      for col in self.continous_columns:
        # nan_indices = input_df[col].index[input_df[col].isna()]
        # input_df.loc[nan_indices, col] = [generate_random_spike(0,100) for _ in range(len(nan_indices))]
        nan_indices = input_df[col].index[input_df[col].isna()]
        df_nonNA = input_df[input_df[col].notna()]

        nonNAN_values = list(df_nonNA[col]) # values of nonNAN
        

        # binning the values
        NUM_BINS = 10000  
        bins = pd.cut(nonNAN_values, NUM_BINS, labels = list(range(0, NUM_BINS)))

  
        bins_frequency = {}
        for item in bins:
          for i in range(0, NUM_BINS):
            if item == i:
              if i in bins_frequency:
                bins_frequency[i] += 1
              else:
                bins_frequency[i] = 1
              break

        # random selection of bins
        population = OrderedSet(bins)
        weights = [v/len(bins) for k, v in bins_frequency.items()]
        k = len(nan_indices)

        selected_bins = random.choices(population, weights, k=k)

        # make a dictionary where the bin numbers are the dictionary keys and the values
        # of each bin is listed as the dictionary's value.
        dict_bins = {}

        for item in zip(bins, nonNAN_values):
          if item[0] in dict_bins:
            dict_bins[item[0]].append(int(item[1]))
          else:
            dict_bins[item[0]] = [item[1]]

        for i in range(0, len(selected_bins)):
          random_value = random.choices(dict_bins[selected_bins[i]])
          input_df.loc[nan_indices[i], col] = random_value[0]




        
    # Preprocess categorical columns
    if (self.categorical_columns is not None):
      input_df_encoded = self.encoder.transform(input_df[self.categorical_columns])
      input_df_encoded_part = pd.DataFrame(input_df_encoded, columns=self.encoder.get_feature_names_out(self.categorical_columns),index=input_df.index)
      input_df = pd.concat([input_df,input_df_encoded_part],axis=1)
      input_df.drop(columns=self.categorical_columns, inplace=True)

      # Handle NaN in categorical columns
      nan_columns = [col for col in self.categorical_columns if '_nan' in col]
      input_df.drop(columns=nan_columns, inplace=True)

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