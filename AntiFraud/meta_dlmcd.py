from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys, os, psutil
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
import gc
from sklearn.metrics import roc_auc_score

import progressbar

import os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not p in sys.path:
    sys.path.append(p)

import utils
import pandas as pd
import config
import DLMCDDataReader
import tensorflow as tf
import dlmcd_utils

process = psutil.Process(os.getpid())
tf.set_random_seed(config.RANDOM_SEED)
pd.set_option('display.max_rows', None)

def _print_memory_usage():
    ''''''
    print('\n---- Current memory usage %sM ----\n' % int(process.memory_info().rss/(1024*1024)))

def _load_data():
    ''''''
    valid_dfs = []
    for fold in range(config.KFOLD):
        FoldInputDir = '%s/kfold/%s' % (config.MetaModelInputDir, fold)
        valid = utils.hdf_loader('%s/valid_label.hdf' % FoldInputDir, 'valid_label').sample(frac= 0.05).reset_index(drop= True)
        valid['fold'] = fold
        valid_dfs.append(valid)
        #if(fold == 0):
            #TestData = pd.read_csv('%s/test.csv' % FoldInputDir)
            #TestData = utils.hdf_loader('%s/test.hdf' % FoldInputDir, 'test').reset_index(drop= True)
    TrainData = pd.concat(valid_dfs, axis= 0, ignore_index= True)

    return TrainData#, TestData

## load
with utils.timer('Loader'):
    _print_memory_usage()
    train_data = _load_data()
    _print_memory_usage()

valid_data = train_data[train_data['fold'] == 4].copy()
train_data = train_data[train_data['fold'] != 4].copy()
## parser
cate_cols = [c for c in train_data.columns if (c.startswith('cate_'))]
fd = DLMCDDataReader.FeatureDictionary(dfTrain= train_data[cate_cols], dfTest= valid_data[cate_cols],ignore_cols=config.IGNORE_COLS)
parser = DLMCDDataReader.DataParser(fd)
# transform
Xi_train, Xv_train, y_train = parser.parse(df= train_data[['label'] + cate_cols], has_label= True)
Xi_valid, Xv_valid, y_valid = parser.parse(df= valid_data[['label'] + cate_cols], has_label= True)
y_train = np.reshape(np.array(y_train), [-1])
y_valid = np.reshape(np.array(y_valid), [-1])
train_data = dlmcd_utils.libsvm_2_coo(zip(Xi_train, Xv_train), (len(Xi_train), fd.feat_dim)).tocsr(), y_train
valid_data = dlmcd_utils.libsvm_2_coo(zip(Xi_valid, Xv_valid), (len(Xi_valid), fd.feat_dim)).tocsr(), y_valid
train_data = dlmcd_utils.shuffle(train_data)
del Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid
gc.collect()

from DLMCD import LR, FM, PNN1, PNN2, FNN, CCPM

input_dim = fd.feat_dim

if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')
print('train data size:', train_data[0].shape)
print('valid data size:', valid_data[0].shape)

train_size = train_data[0].shape[0]
valid_size = valid_data[0].shape[0]
num_feas = len(fd.field_size)

min_round = 1
num_round = 200
early_stop_round = 5
batch_size = 256

field_sizes = fd.field_size
field_offsets = fd.field_offset

algo = 'pnn2'

if algo in {'fnn', 'ccpm', 'pnn1', 'pnn2'}:
    train_data = dlmcd_utils.split_data(train_data, fd.field_offset)
    valid_data = dlmcd_utils.split_data(valid_data, fd.field_offset)
    tmp = []
    for x in field_sizes:
        if x > 0:
            tmp.append(x)
    field_sizes = tmp
    print('remove empty fields', field_sizes)

if algo == 'lr':
    lr_params = {
        'input_dim': input_dim,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_weight': 0,
        'random_seed': 0
    }
    print(lr_params)
    model = LR(**lr_params)
elif algo == 'fm':
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 10,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_w': 0,
        'l2_v': 0,
    }
    print(fm_params)
    model = FM(**fm_params)
elif algo == 'fnn':
    fnn_params = {
        'field_sizes': field_sizes,
        'embed_size': 10,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0, 0],
        'random_seed': 0
    }
    print(fnn_params)
    model = FNN(**fnn_params)
elif algo == 'ccpm':
    ccpm_params = {
        'field_sizes': field_sizes,
        'embed_size': 10,
        'filter_sizes': [5, 3],
        'layer_acts': ['relu'],
        'drop_out': [0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'random_seed': 0
    }
    print(ccpm_params)
    model = CCPM(**ccpm_params)
elif algo == 'pnn1':
    pnn1_params = {
        'field_sizes': field_sizes,
        'embed_size': 10,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0, 0],
        'random_seed': 0
    }
    print(pnn1_params)
    model = PNN1(**pnn1_params)
elif algo == 'pnn2':
    # pnn2_params = {
    #     'field_sizes': field_sizes,
    #     'embed_size': 10,
    #     'layer_sizes': [500, 1],
    #     'layer_acts': ['relu', None],
    #     'drop_out': [0, 0],
    #     'opt_algo': 'gd',
    #     'learning_rate': 0.1,
    #     'embed_l2': 0,
    #     'layer_l2': [0., 0.],
    #     'random_seed': 0,
    #     'layer_norm': True,
    # }
    pnn2_params = {
        'field_sizes': field_sizes,
        'embed_size': 8,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0., 0.],
        'random_seed': 0,
        'layer_norm': True,
    }
    print(pnn2_params)
    model = PNN2(**pnn2_params)


def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                X_i, y_i = dlmcd_utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = dlmcd_utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = []
        print('[%d]\tevaluating...' % i)
        bar = progressbar.ProgressBar()
        for j in bar(range(int(train_size / 10000 + 1))):
            X_i, _ = dlmcd_utils.slice(train_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            train_preds.extend(preds)
        test_preds = []
        bar = progressbar.ProgressBar()
        for j in bar(range(int(valid_size / 10000 + 1))):
            X_i, _ = dlmcd_utils.slice(valid_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            test_preds.extend(preds)
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(valid_data[1], test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break

train(model)
