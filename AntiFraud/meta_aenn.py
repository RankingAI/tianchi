import config
import utils
import DataReader
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys,os,psutil
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime, time
import numba
import gc
from AutoEncoder import AutoEncoder

process = psutil.Process(os.getpid())
tf.set_random_seed(config.RANDOM_SEED)
pd.set_option('display.max_rows', None)

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

def _load_data(kfold, inputdir):
    ''''''
    valid_dfs = []
    for fold in range(kfold):
        FoldInputDir = '%s/kfold/%s' % (inputdir, fold)
        if(config.ae_params['debug'] == True):
            sample_frac = 0.1
        else:
            sample_frac = 1.0
        valid = utils.hdf_loader('%s/valid_label.hdf' % FoldInputDir, 'valid_label').sample(frac= sample_frac).reset_index(drop= True)
        valid['fold'] = fold
        valid_dfs.append(valid)
        if(fold == 0):
            TestData = utils.hdf_loader('%s/test.hdf' % FoldInputDir, 'test').sample(frac= sample_frac).reset_index(drop= True)
    TrainData = pd.concat(valid_dfs, axis= 0, ignore_index= True)

    return TrainData, TestData

@numba.jit
def ApplyTransform(col_name, col_vals, lookup_table):
    results = np.zeros(len(col_vals), dtype= 'float32')
    for i in range(len(col_vals)):
        if(col_vals[i] in lookup_table[col_name]):
            results[i] = lookup_table[col_name][col_vals[i]]
    return results

def _run_meta_model_aenn():
    ''''''
    with utils.timer('Load'):
        TrainData, TestData = _load_data(config.KFOLD, config.MetaModelInputDir)

    print('\n------------------------')
    trans_start = time.time()

    ## transform on categorical data with fraud ratio
    cate_feats = [c for c in TrainData.columns if (c.startswith('cate_'))]
    cate_fraud_ratio = {}
    with utils.timer('Statistics for fraud ratio'):
        for c in cate_feats:
            ret = TrainData[[c, 'label']].groupby(c).agg({'label': [sum, len]}).reset_index()
            ret.columns = [c, 'fraud_sum', 'total']
            ret['fraud_ratio'] = ret['fraud_sum'] / ret['total']
            ret.drop(['fraud_sum', 'total'], axis=1, inplace=True)
            cate_fraud_ratio[c] = dict(zip(ret[c].values, ret['fraud_ratio'].values))
    with utils.timer('Transformation on train'):
        for c in cate_feats:
            TrainData[c] = ApplyTransform(c, TrainData[c].values, cate_fraud_ratio)
            TestData[c] = ApplyTransform(c, TestData[c].values, cate_fraud_ratio)
    del cate_fraud_ratio
    gc.collect()

    trans_end = time.time()
    print('Transformation on categorical data done, time elapsed %ss.' % int(trans_end - trans_start))
    print('------------------------\n')

    ## CV for auto encoder
    print('\n-------------------------')
    ae_start = time.time()
    num_feats = [c for c in TrainData.columns if(c.startswith('num_'))]
    feats = num_feats.copy()
    feats.extend(cate_feats)
    feats = [c for c in feats if(c not in config.IGNORE_COLS)]
    sep_idx = len([c for c in feats if(c.startswith('num_'))])
    print('Feature size %s' % len(feats))
    for fold in range(config.KFOLD):
        # get X,y
        FoldTrain = TrainData[TrainData['fold'] != fold].copy()
        X_train, y_train = FoldTrain[feats].values, FoldTrain['label'].values
        del FoldTrain
        gc.collect()
        FoldValid = TrainData[TrainData['fold'] == fold].copy()
        X_valid, y_valid = FoldValid[feats].values, FoldValid['label'].values
        del FoldValid
        gc.collect()
        # # normalization with min-max
        # cols_max = []
        # cols_min = []
        # for c in range(X_train.shape[1]):
        #     cols_max.append(X_train[:, c].max())
        #     cols_min.append(X_train[:, c].min())
        #     X_train[:, c] = (X_train[:, c] - cols_min[-1])/(cols_max[-1] - cols_min[-1])
        #     X_valid[:, c] = (X_valid[:, c] - cols_min[-1])/(cols_max[-1] - cols_min[-1])
        # normalization with z-score
        cols_mean = []
        cols_std = []
        for c in range(X_train.shape[1]):
            if(c < sep_idx):
                X_train[:, c] = (X_train[:, c])
                X_valid[:, c] = (X_valid[:, c])
            cols_mean.append(X_train[:, c].mean())
            cols_std.append(X_train[:, c].std())
            X_train[:, c] = (X_train[:, c] - cols_mean[-1]) / cols_std[-1]
            X_valid[:, c] = (X_valid[:, c] - cols_mean[-1]) / cols_std[-1]
        # Parameters
        ae_params['feature_size'] = len(feats)
        ae_params['model_path'] = '%s/ae_model' % config.MetaModelOutputDir
        if(os.path.exists(ae_params['model_path']) == False):
            os.makedirs(ae_params['model_path'])
        # train/inference
        ae = AutoEncoder(**ae_params)
        ae.fit(X_train, y_train, X_valid, y_valid)
        break

    ae_end = time.time()
    print('Auto encoder done, time elapsed %ss.' % int(ae_end - ae_start))
    print('------------------------\n')

if __name__ == '__main__':
    ''''''
    _run_meta_model_aenn()
