import pandas as pd
import numpy as np
from AutoEncoder import AutoEncoder
import config

df = pd.read_csv('%s/raw/creditcard.csv' % config.DataBaseDir)

ae_params = {
    'feature_size': 0,
    'encoder_layers': config.ae_params['encoder_layers'],
    'learning_rate': config.ae_params['learning_rate'],
    'epochs': config.ae_params['epochs'],
    'batch_size': config.ae_params['batch_size'],
    'random_seed': config.ae_params['random_seed'],
    'display_step': config.ae_params['display_step'],
    'verbose': config.ae_params['verbose'],
    'model_path': '.',
}

TEST_RATIO = 0.25
df.sort_values('Time', inplace = True)
TRA_INDEX = int((1-TEST_RATIO) * df.shape[0])
train_x = df.iloc[:TRA_INDEX, 1:-2].values
train_y = df.iloc[:TRA_INDEX, -1].values

test_x = df.iloc[TRA_INDEX:, 1:-2].values
test_y = df.iloc[TRA_INDEX:, -1].values

cols_mean = []
cols_std = []
for c in range(train_x.shape[1]):
    cols_mean.append(train_x[:,c].mean())
    cols_std.append(train_x[:,c].std())
    train_x[:, c] = (train_x[:, c] - cols_mean[-1]) / cols_std[-1]
    test_x[:, c] =  (test_x[:, c] - cols_mean[-1]) / cols_std[-1]

ae_params['feature_size'] = 28
ae_params['model_path'] = '%s/ae_model' % config.MetaModelOutputDir
ae = AutoEncoder(**ae_params)
ae.fit(train_x, train_y, test_x, test_y)
