#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split

import azureml.core
from azureml.core import Workspace
from azureml.core.dataset import Dataset

# The Grab parquet file from Azure Blob storage 
filepath = "https://grab5033896937.blob.core.windows.net/azureml/Dataset/grab/part-00000-41e3fe2a-7fad-41df-aba6-d805d478bc9f-c000.snappy.parquet"
RANDOM_SEED = 69

# create a TabularDataset from a delimited file behind a public web url
df = Dataset.Tabular.from_parquet_files(path=filepath)

# convert to pure df
df = df.to_pandas_dataframe()

# chcck the total size 
print(len(df))

df = df.set_index('trj_id')
df = df.loc[:, ~df.columns.isin(['start_dt', 'end_dt', 'bearing_diff', 'avg_accuracy'])]
df = df.rename(columns={'time_diff': 'eta'})

# split between labels and features
X = df.loc[:, df.columns != 'eta']
y = df.loc[:, df.columns == 'eta']

# split into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# hyper-parameter specification
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['rmse'],
    'learning_rate': 0.001,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 10,
    "num_leaves": 40,
    "max_bin": 1024,
    "num_iterations": 100000,
    "n_estimators": 100,
    "seed": RANDOM_SEED
}

#Train a regression model with early_stopping_rounds of 1000
gbm = lgb.LGBMRegressor(**hyper_params)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        early_stopping_rounds=1000)

# create "outputs" directory
os.makedirs('outputs', exist_ok=True)

# The training script saves the model into a directory named ‘outputs’. Note files saved in the
# outputs folder are automatically uploaded into experiment record. Anything written in this
# directory is automatically uploaded into the workspace.
pkl_filename = "outputs/lightgbm.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(gbm, file)
