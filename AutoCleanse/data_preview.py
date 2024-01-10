import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline

df = pd.read_csv('~/dataset/categorical/adult.csv').drop(columns=['fnlwgt'])
og_columns = df.columns.to_list()
# continous_columns = ['age','hours.per.week']
# categorical_columns = ['workclass','education','education.num','marital.status','occupation','relationship','race','sex','native.country','income']
X,y = df.drop('income',axis=1), df['income']
continous_columns = X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X.select_dtypes(include=['object', 'bool']).columns
print(y.dtype)
# log_transformer = FunctionTransformer(np.log1p, validate=True)
# data = df['age']
# pipe1 = make_pipeline(log_transformer,MinMaxScaler())
# data_proc1 = pipe1.fit_transform(data.values.reshape(-1, 1))
# pipe2 = make_pipeline(MinMaxScaler())
# data_proc2 = pipe2.fit_transform(data.values.reshape(-1, 1))

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# axes[0].hist(data, bins=20, color='blue', alpha=0.7)
# axes[0].set_title('Original')
# axes[0].set_xlabel('Value')
# axes[0].set_ylabel('Count')

# axes[1].hist(data_proc1, bins=20, color='green', alpha=0.7)
# axes[1].set_title('Preprocessed 1')
# axes[1].set_xlabel('Value')
# axes[1].set_ylabel('Count')

# axes[2].hist(data_proc2, bins=20, color='red', alpha=0.7)
# axes[2].set_title('Preprocessed 2')
# axes[2].set_xlabel('Value')
# axes[2].set_ylabel('Count')

# for ax in axes:
#     ax.grid(axis='y', alpha=0.75)

# plt.tight_layout()
# plt.show()