import pandas as pd
import numpy as np

import sys, os, time, datetime
import numpy as np
import pandas as pd
import config
import utils
from sklearn.model_selection import StratifiedKFold
import lightgbm
from sklearn.metrics import roc_auc_score
import gc

sys.path.append("..")
pd.set_option('display.max_rows', None)

num_iterations =  5000
params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    #'metric': 'binary_logloss',
    'metric': 'None',

    #'num_iterations': 5000,
    'learning_rate': 0.01,  # !!!
    'num_leaves': 63,
    'max_depth': 8,  # !!!
    'scale_pos_weight': 1,
    'verbose': -10,

    'feature_fraction': 0.6,
    'bagging_fraction': 0.9,

    'max_bin': 255,
}
strategy = 'lgb_sss'
debug = False
pl_sampling_rate = 0.05
pl_sample_weight = 0.01
pl_sampling_times = 1
train_times = 16
unlabeled_weight = 1.0

## loading data
cal_dtypes = {
    'dow': 'uint8',
    'is_holiday': 'uint8'
}
dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
with utils.timer('Load data'):
    DataSet = {
        'train': pd.read_csv('%s/raw/atec_anti_fraud_train.csv' % config.DataRootDir, parse_dates=['date'],
                             date_parser=dateparse),
        # 'test': pd.read_csv('%s/raw/atec_anti_fraud_test_b.csv' % config.DataRootDir, parse_dates=['date'],
        #                     date_parser=dateparse),
        'calendar': pd.read_csv('%s/raw/calendar.csv' % config.DataRootDir, parse_dates=['date'],
                                date_parser=dateparse, dtype=cal_dtypes, usecols=['date', 'dow', 'is_holiday'])
    }
    if(debug == True):
        tra_idx_list = [v for v in DataSet['train'].index.values if ((v % 10) == 0)]
        DataSet['train'] = DataSet['train'].iloc[tra_idx_list, :].reset_index(drop= True)
        # tes_idx_list = [v for v in DataSet['test'].index.values if ((v % 10) == 0)]
        # DataSet['test'] = DataSet['test'].iloc[tes_idx_list, :].reset_index(drop= True)
    for mod in ['calendar', 'train']:
        if (mod == 'calendar'):
            DataSet[mod]['dow'] -= 1
        DataSet[mod]['date'] = DataSet[mod]['date'].dt.date
        DataSet[mod].sort_values(by=['date'], inplace=True)

## add missing feature number
with utils.timer('Add missing feat number'):
    for mod in ['train']:
        DataSet[mod]['num_missing_feat'] = DataSet[mod][config.NUMERIC_COLS + config.CATEGORICAL_COLS].isnull().sum(axis=1)

## add week no for data split
with utils.timer('Add week no'):
    DataSet['calendar']['wno'] = (DataSet['calendar']['date'].apply(lambda  x: (x - datetime.datetime(2017, 9, 5).date()).days) / 7).astype('int16')

## renaming
with utils.timer('Rename columns'):
    date_col_map = dict((col, 'date_%s' % col) for col in DataSet['calendar'].columns if (col not in ['date', 'wno']))
    DataSet['calendar'].rename(index=str, columns= date_col_map, inplace=True)

## merge date data
with utils.timer('Merge date data'):
    for mod in ['train']:
        DataSet[mod] = DataSet[mod].merge(DataSet['calendar'], on=['date'], how='left')

print('\n ======== Summary ======== ')
## label
uni, count = np.unique(DataSet['train']['label'], return_counts=True)
label_count = dict(zip(uni, count))
print('Empty label rate %.3f' % (label_count[-1] / np.sum(count)))
print('Positive label rate %.3f' % (label_count[1] / np.sum(count)))
print('Negative label rate %.3f' % (label_count[0] / np.sum(count)))

print('------------------------------')
## size/len
for mod in ['train']:
    data_len = len(DataSet[mod])
    data_size = DataSet[mod].size
    print('%s: data len %s, data size %.2fM' % (mod, data_len, (data_size / 1e6)))
print('------------------------------')

## date
for mod in ['train']:
    start_date, end_date = DataSet[mod]['date'].min(), DataSet[mod]['date'].max()
    print('%s: date range %s - %s' % (mod, start_date, end_date))
print('=======================\n')

## treat the whole raw features as the numeric ones
raw_cols = [c for c in DataSet['train'].columns if(c.startswith('f'))]
date_cols = [c for c in DataSet['train'].columns if(c.startswith('date_'))]
# drop_cols = ["f24", "f26", "f27", "f28", "f29", "f30", "f31", "f34", "f52", "f53", "f54", "f58"]
drop_cols = ["f34","f54","f58","f106","f81","f99","f28","f29","f30","f31","f24","f25","f26","f27","f21","f52","f53"]
raw_cols = [c for c in raw_cols if(c not in drop_cols)]
total_feat_cols = raw_cols

def evalauc(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', roc_auc_score(labels, [1 if(v > 0.5) else 0 for v in preds]), True

def evaltpr(preds, dtrain):
    labels = dtrain.get_label()
    return 'tpr', utils.sum_weighted_tpr(labels, preds), True

def proba2label(data):
    return [1 if(v > 0.5) else 0 for v in data]

weeks = np.max(DataSet['train']['wno']) + 1

labeled_index = DataSet['train'].index[DataSet['train']['label'] != -1]

eval_index = DataSet['train'].index[(DataSet['train']['wno'] >= weeks - 2) & (DataSet['train']['label'] != -1)]
del DataSet
gc.collect()

#sys.exit(1)
DataBaseDir = '/Users/yuanpingzhou/project/workspace/python/tianchi/AntiFraud/data/7'
data = pd.read_csv('%s/lgb_sss_te_cv_train_0.339762.csv' % DataBaseDir)
cv_train_tpr = utils.sum_weighted_tpr(data['label'].iloc[eval_index,], data['score'].iloc[eval_index,])
whole_cv_train_tpr = utils.sum_weighted_tpr(data['label'].iloc[labeled_index,], data['score'].iloc[labeled_index,])

pre_data = pd.read_csv('%s/lgb_sss_te_pre_cv_train_0.343747.csv' % DataBaseDir)
pre_cv_train_tpr = utils.sum_weighted_tpr(pre_data['label'].iloc[eval_index,], pre_data['score'].iloc[eval_index,])
whole_pre_cv_train_tpr = utils.sum_weighted_tpr(pre_data['label'].iloc[labeled_index,], pre_data['score'].iloc[labeled_index,])

print(cv_train_tpr, pre_cv_train_tpr)
print(whole_cv_train_tpr, whole_pre_cv_train_tpr)

DataBaseDir = '/Users/yuanpingzhou/project/workspace/python/tianchi/AntiFraud/data/5'
data = pd.read_csv('%s/lgb_sss_te_cv_train_0.349802.csv' % DataBaseDir)
cv_train_tpr = utils.sum_weighted_tpr(data['label'].iloc[eval_index,], data['score'].iloc[eval_index,])
whole_cv_train_tpr = utils.sum_weighted_tpr(data['label'].iloc[labeled_index,], data['score'].iloc[labeled_index,])

pre_data = pd.read_csv('%s/lgb_sss_te_pre_cv_train_0.385588.csv' % DataBaseDir)
pre_cv_train_tpr = utils.sum_weighted_tpr(pre_data['label'].iloc[eval_index,], pre_data['score'].iloc[eval_index,])
whole_pre_cv_train_tpr = utils.sum_weighted_tpr(pre_data['label'].iloc[labeled_index,], pre_data['score'].iloc[labeled_index,])

print(cv_train_tpr, pre_cv_train_tpr)
print(whole_cv_train_tpr, whole_pre_cv_train_tpr)