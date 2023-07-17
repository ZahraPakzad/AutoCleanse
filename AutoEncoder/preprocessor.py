import pandas as pd
from sklearn.model_selection import train_test_split

def dataPreprocessor(input_df,train_ratio,val_ratio,test_ratio,continous_columns,categorical_columns,scaler,onehotencoder):
    """
     @brief Split dataset into training, validation and test set, then apply scaling and encoding to the columns
     @param input_df: Input dataframe
     @param train_ratio: float or int ratio of training data
     @param val_ratio: float or int ratio of validation data
     @param test_ratio: float or int ratio of test data
     @param continous_columns: A list of continous column names
     @param categorical_columns: A list of categorical column names
     @param scaler: Scaler object
     @param onehotencoder: Onehot encoder object
    """
    # Calculate the sizes of train, validation, and test sets
    total_size = len(input_df)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split training, validation, and test sets
    X_train, X_temp = train_test_split(input_df, test_size=(val_size + test_size), random_state=42)  #@TODO: random seed   
    X_val, X_test = train_test_split(X_temp, test_size=test_size, random_state=42)

    # Preprocess continous columns
    X_train_scaled = scaler.fit_transform(X_train[continous_columns])
    X_val_scaled = scaler.transform(X_val[continous_columns])
    X_test_scaled = scaler.transform(X_test[continous_columns])
    X_train[continous_columns] = X_train_scaled
    X_val[continous_columns] = X_val_scaled
    X_test[continous_columns] = X_test_scaled

    # Preprocess categorical columns
    X_train_encoded = onehotencoder.fit_transform(X_train[categorical_columns])
    X_train_encoded_df_part = pd.DataFrame(X_train_encoded, columns=onehotencoder.get_feature_names_out(categorical_columns),index=X_train.index)
    X_train = pd.concat([X_train,X_train_encoded_df_part],axis=1)
    X_train.drop(columns=categorical_columns, inplace=True)

    X_val_encoded = onehotencoder.transform(X_val[categorical_columns])
    X_val_encoded_df_part = pd.DataFrame(X_val_encoded, columns=onehotencoder.get_feature_names_out(categorical_columns),index=X_val.index)
    X_val = pd.concat([X_val,X_val_encoded_df_part],axis=1)
    X_val.drop(columns=categorical_columns, inplace=True)

    X_test_encoded = onehotencoder.transform(X_test[categorical_columns])
    X_test_encoded_df_part = pd.DataFrame(X_test_encoded, columns=onehotencoder.get_feature_names_out(categorical_columns),index=X_test.index)
    X_test = pd.concat([X_test,X_test_encoded_df_part],axis=1)
    X_test.drop(columns=categorical_columns, inplace=True)

    return X_train, X_val, X_test