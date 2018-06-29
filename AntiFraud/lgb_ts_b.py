import sys, os, time, datetime
import numpy as np
import pandas as pd
import config
import utils
from sklearn.model_selection import StratifiedKFold
import lightgbm
from sklearn.metrics import roc_auc_score
from itertools import combinations

sys.path.append("..")
pd.set_option('display.max_rows', None)

num_iterations =  5000
params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'None',

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
unlabeled_weight = 0.8

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
        'test': pd.read_csv('%s/raw/atec_anti_fraud_test_b.csv' % config.DataRootDir, parse_dates=['date'],
                            date_parser=dateparse),
        'calendar': pd.read_csv('%s/raw/calendar.csv' % config.DataRootDir, parse_dates=['date'],
                                date_parser=dateparse, dtype=cal_dtypes, usecols=['date', 'dow', 'is_holiday'])
    }
    if(debug == True):
        tra_idx_list = [v for v in DataSet['train'].index.values if ((v % 10) == 0)]
        DataSet['train'] = DataSet['train'].iloc[tra_idx_list, :].reset_index(drop= True)
        tes_idx_list = [v for v in DataSet['test'].index.values if ((v % 10) == 0)]
        DataSet['test'] = DataSet['test'].iloc[tes_idx_list, :].reset_index(drop= True)
    for mod in ['calendar', 'train', 'test']:
        if (mod == 'calendar'):
            DataSet[mod]['dow'] -= 1
        DataSet[mod]['date'] = DataSet[mod]['date'].dt.date
        DataSet[mod].sort_values(by=['date'], inplace=True)

## add missing feature number
with utils.timer('Add missing feat number'):
    for mod in ['train', 'test']:
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
    for mod in ['train', 'test']:
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
for mod in ['train', 'test']:
    data_len = len(DataSet[mod])
    data_size = DataSet[mod].size
    print('%s: data len %s, data size %.2fM' % (mod, data_len, (data_size / 1e6)))
print('------------------------------')

## date
for mod in ['train', 'test']:
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

def train_with_cv(train_data, test_data, kfold, skf, cv_params, s, w, weights):
    ''''''
    best_trees = []
    cv_pred_train = np.zeros(len(train_data))
    cv_pred_test = np.zeros(len(test_data))
    cv_tpr_scores = np.zeros(kfold)
    cv_auc_scores = np.zeros(kfold)

    kf = skf.split(train_data[total_feat_cols], train_data['label'])

    for fold, (train_index, valid_index) in enumerate(kf):
        X_train, X_valid = train_data[total_feat_cols].iloc[train_index,].reset_index(drop=True), train_data[total_feat_cols].iloc[valid_index,].reset_index(drop=True)
        y_train, y_valid = train_data['label'].iloc[train_index,].reset_index(drop=True), train_data['label'].iloc[valid_index,].reset_index(drop=True)
        w_train, w_valid = weights[train_index], weights[valid_index]

        dtrain = lightgbm.Dataset(X_train, y_train, weight= w_train, feature_name=total_feat_cols)  # , categorical_feature= date_cols)#likely_cate_cols)
        dvalid = lightgbm.Dataset(X_valid, y_valid, weight= w_valid, reference=dtrain,feature_name=total_feat_cols)  # , categorical_feature= date_cols)#likely_cate_cols)

        bst = lightgbm.train(cv_params, dtrain, num_iterations, valid_sets=dvalid, feval= evaltpr, verbose_eval=50,early_stopping_rounds=100)
        best_trees.append(bst.best_iteration)

        cv_pred_test += bst.predict(test_data[total_feat_cols], num_iteration=bst.best_iteration)  # * week_weight
        cv_pred_train[valid_index] += bst.predict(X_valid, num_iteration=bst.best_iteration)  # * week_weight

        label_positives = np.sum(y_valid)
        pred_positives = (np.sum([1.0 if (v > 0.5) else 0 for v in cv_pred_train[valid_index]]))

        cv_tpr_scores[fold] = utils.sum_weighted_tpr(y_valid, cv_pred_train[valid_index])
        cv_auc_scores[fold] = roc_auc_score(y_valid, [1 if (v > 0.5) else 0 for v in cv_pred_train[valid_index]])

        print('\n---------------------------------------------------')
        print('#%s: week %s, fold %s, score %.6f/%.6f, positives %s/%s' % (s, w, fold,
                                                                           cv_tpr_scores[fold],cv_auc_scores[fold],
                                                                           label_positives,pred_positives))
        print('---------------------------------------------------\n')

    cv_pred_test /= kfold

    return cv_pred_train, cv_pred_test, cv_tpr_scores, cv_auc_scores

def pseudo_label_sampling(proba_array, sample_rate, pos_ratio, neg_ratio):
    sorted_preds = np.sort(proba_array, axis=None)
    down_thre, up_thre = sorted_preds[:int(0.983 * sample_rate * neg_ratio * len(sorted_preds))][-1], sorted_preds[-int(0.012 * sample_rate * pos_ratio * len(sorted_preds)):][0]
    pl_sample_index = np.where((proba_array >= up_thre) | (proba_array <= down_thre))[0]
    remained_index = np.where((proba_array < up_thre) & (proba_array > down_thre))[0]

    print(down_thre, up_thre)

    return pl_sample_index, remained_index

def public_train(data, weeks):
    data['test']['label'] = 0
    wno2 = 4

    valid_index = data['train'].index[data['train']['wno2'] == wno2 - 1]
    valid_data = data['train'].iloc[valid_index,].reset_index(drop=True)
    val_labeled_index = valid_data.index[valid_data['label'] != -1]

    ##
    final_pre_cv_pred_valid = np.zeros(len(valid_index))
    final_cv_pred_valid = np.zeros(len(valid_index))
    final_pre_cv_pred_test = np.zeros(len(data['test']))
    final_cv_pred_test = np.zeros(len(data['test']))

    ##
    pre_entire_cv_tpr_scores = np.zeros((train_times, wno2))
    entire_cv_tpr_scores = np.zeros((train_times, wno2))

    pre_cv_tpr_scores = np.zeros((train_times, wno2))
    cv_tpr_scores = np.zeros((train_times, wno2))

    ##
    pre_entire_t_cv_tpr_scores = np.zeros(train_times)
    pre_entire_t_agg_cv_tpr_scores = np.zeros(train_times)
    entire_t_cv_tpr_scores = np.zeros(train_times)
    entire_t_agg_cv_tpr_scores = np.zeros(train_times)

    pre_t_cv_tpr_scores = np.zeros(train_times)
    pre_t_agg_cv_tpr_scores = np.zeros(train_times)
    t_cv_tpr_scores = np.zeros(train_times)
    t_agg_cv_tpr_scores = np.zeros(train_times)

    start = time.time()

    data['train']['wno2'] = 0
    data['train'].loc[data['train']['wno']/2 < 1, 'wno2'] = 0
    data['train'].loc[(data['train']['wno']/2 >= 1) & (data['train']['wno']/2 <= 2), 'wno2'] = 1
    data['train'].loc[(data['train']['wno']/2 > 2) & (data['train']['wno']/2 <= 3), 'wno2'] = 2
    data['train'].loc[data['train']['wno']/2 > 3, 'wno2'] = 3
    data['test']['wno2'] = 0

    print('\n------------- DEBUG ----------------')
    print(data['train']['wno2'].value_counts())
    print('------------------------------------\n')


    for s in range(train_times):
        params['seed'] = s

        pre_cv_pred_test = np.zeros(len(data['test']))
        pre_cv_pred_valid = np.zeros(len(valid_index))

        cv_pred_test = np.zeros(len(data['test']))
        cv_pred_valid = np.zeros(len(valid_index))

        ## pre-train
        for w2 in range(wno2 - 1):

            train_index = data['train'].index[data['train']['wno2'] != w2]
            train_data = data['train'].iloc[train_index,].reset_index(drop= True)
            tra_unlabeled_index, val_unlabeled_index = train_data.index[train_data['label'] == -1], valid_data.index[valid_data['label'] == -1]
            train_data.loc[train_data['label'] == -1, 'label'] = 1
            valid_data.loc[valid_data['label'] == -1, 'label'] = 1

            w_train = np.ones(len(train_data)).astype('float32')
            w_train[tra_unlabeled_index] = unlabeled_weight
            w_valid = np.ones(len(valid_data)).astype('float32')
            w_valid[val_unlabeled_index] = unlabeled_weight

            dtrain = lightgbm.Dataset(train_data[total_feat_cols], train_data['label'], weight=w_train, feature_name=total_feat_cols)
            dvalid = lightgbm.Dataset(valid_data[total_feat_cols], valid_data['label'], weight= w_valid, reference=dtrain, feature_name=total_feat_cols)

            bst = lightgbm.train(params, dtrain, num_iterations, valid_sets=dvalid, feval=evaltpr, verbose_eval=50,early_stopping_rounds=100)

            pre_cv_pred_test += bst.predict(data['test'][total_feat_cols], num_iteration=bst.best_iteration)
            pre_cv_pred_valid += bst.predict(valid_data[total_feat_cols], num_iteration=bst.best_iteration)

            ## pl sampling
            pl_sample_index, remained_index = pseudo_label_sampling(pre_cv_pred_test, pl_sampling_rate, 0.5, 0.5)
            pl_data = data['test'][train_data.columns].iloc[pl_sample_index,].reset_index(drop=True)
            pl_data['label'] = np.array(proba2label(pre_cv_pred_test[pl_sample_index]))

            ## re-train
            post_train_data = pd.concat([train_data, pl_data], axis= 0, ignore_index= True)
            post_w_train = np.array(list(w_train) + list(np.full(len(pl_data), pl_sample_weight)))

            dtrain = lightgbm.Dataset(post_train_data[total_feat_cols], post_train_data['label'], weight= post_w_train, feature_name=total_feat_cols)
            dvalid = lightgbm.Dataset(valid_data[total_feat_cols], valid_data['label'], weight= w_valid, reference=dtrain, feature_name=total_feat_cols)

            bst = lightgbm.train(params, dtrain, num_iterations, valid_sets=dvalid, feval=evaltpr, verbose_eval=50,early_stopping_rounds=100)

            cv_pred_test[remained_index] += bst.predict(data['test'][total_feat_cols].iloc[remained_index,], num_iteration=bst.best_iteration)
            cv_pred_test[pl_sample_index] += bst.predict(train_data[total_feat_cols][len(train_index):,], num_iteration= bst.best_iteration)
            cv_pred_valid += bst.predict(valid_data[total_feat_cols], num_iteration=bst.best_iteration)

            pre_cv_tpr_scores[s][w2] = utils.sum_weighted_tpr(valid_data['label'].iloc[val_labeled_index,], pre_cv_pred_valid[val_labeled_index])
            cv_tpr_scores[s][w2] = utils.sum_weighted_tpr(valid_data['label'].iloc[val_labeled_index,], cv_pred_valid[val_labeled_index])

            pre_entire_cv_tpr_scores[s][w2] = utils.sum_weighted_tpr(valid_data['label'], pre_cv_pred_valid)
            entire_cv_tpr_scores[s][w2] = utils.sum_weighted_tpr(valid_data['label'], cv_pred_valid)

            print('\n------------------------------')
            print('entire valid score %.6f/%.6f, labeled valid score %.6f/%.6f' % (pre_entire_cv_tpr_scores[s][w2], entire_cv_tpr_scores[s][w2],
                                                                                   pre_cv_tpr_scores[s][w2], cv_tpr_scores[s][w2]))
            print('original positives %s, pl positives %s' % (np.sum(train_data['label']), np.sum(pl_data['label'])))
            print('------------------------------\n')

        pre_cv_pred_valid /= (wno2 - 1)
        pre_cv_pred_test /= (wno2 - 1)
        cv_pred_valid /= (wno2 - 1)
        cv_pred_test /= (wno2 - 1)

        final_cv_pred_valid += cv_pred_valid
        final_pre_cv_pred_valid += pre_cv_pred_valid
        final_cv_pred_test += cv_pred_test
        final_pre_cv_pred_test += pre_cv_pred_test

        t_cv_tpr_scores[s] = utils.sum_weighted_tpr(valid_data['label'].iloc[val_labeled_index,], cv_pred_valid[val_labeled_index])
        t_agg_cv_tpr_scores[s] = utils.sum_weighted_tpr(val_labeled_index['label'].iloc[val_labeled_index,], final_cv_pred_valid[val_labeled_index]/(s + 1.0))

        pre_t_cv_tpr_scores[s] = utils.sum_weighted_tpr(valid_data['label'].iloc[val_labeled_index,], pre_cv_pred_valid[val_labeled_index])
        pre_t_agg_cv_tpr_scores[s] = utils.sum_weighted_tpr(val_labeled_index['label'].iloc[val_labeled_index,], final_pre_cv_pred_valid[val_labeled_index]/(s + 1.0))

        entire_t_cv_tpr_scores[s] = utils.sum_weighted_tpr(valid_data['label'], cv_pred_valid)
        entire_t_agg_cv_tpr_scores[s] = utils.sum_weighted_tpr(val_labeled_index['label'], final_cv_pred_valid/(s + 1.0))

        pre_entire_t_cv_tpr_scores[s] = utils.sum_weighted_tpr(valid_data['label'], pre_cv_pred_valid)
        pre_entire_t_agg_cv_tpr_scores[s] = utils.sum_weighted_tpr(val_labeled_index['label'], final_pre_cv_pred_valid/(s + 1.0))

        pre_pred_positives = np.sum(proba2label(pre_cv_pred_test))
        pred_positives = np.sum(proba2label(cv_pred_test))

        print('\n===============================')
        print('labeled valid: ')
        print('current time scores %.6f/%.6f, aggregated time scores %.6f/%.6f' % (pre_t_cv_tpr_scores[s], t_cv_tpr_scores[s],
                                                                                   pre_t_agg_cv_tpr_scores[s], t_agg_cv_tpr_scores[s]))
        print('entire valid: ')
        print('current time scores %.6f/%.6f, aggregated time scores %.6f/%.6f' % (pre_entire_t_cv_tpr_scores[s], entire_t_cv_tpr_scores[s],
                                                                                   pre_entire_t_agg_cv_tpr_scores[s], entire_t_agg_cv_tpr_scores[s]))
        print('predict positives: ')
        print('%s/%s' % (pre_pred_positives, pred_positives))
        print('===============================\n')
        ##
