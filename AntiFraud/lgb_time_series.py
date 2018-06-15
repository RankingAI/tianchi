import sys, os, time, datetime
import numpy as np
import pandas as pd
import config
import utils
from sklearn.model_selection import StratifiedKFold
import lightgbm

sys.path.append("..")
pd.set_option('display.max_rows', None)

params = {
    "boosting": "gbdt",
    "objective": "binary",
    'metric': "None",
    #"lambda_l2": 2,  # !!!

    "num_iterations": 5000,
    "learning_rate": 0.1,  # !!!
    "max_depth": 8,  # !!!
    #'scale_pos_weight': 5,
    #'min_data_in_leaf': 2000,
    #'min_child_samples': 50,
    #'min_child_weight': 150,
    'min_split_gain': 0,
    'num_leaves': 255,
    #'cat_smooth': 80,
    #'cat_l2': 20,
    #'drop_rate': 0.1,
    #'max_drop': 50,
    #'max_cat_threshold': 64,
    'verbose': -10,

    "feature_fraction": 0.6,
    "bagging_fraction": 0.9,

    "max_bin": 255,
}

strategy = 'lgb_time_series'
debug = False

start = time.time()
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
        'test': pd.read_csv('%s/raw/atec_anti_fraud_test_a.csv' % config.DataRootDir, parse_dates=['date'],
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

## add date features
with utils.timer('Add date features'):
    # add is_weekend
    DataSet['calendar']['is_weekend'] = DataSet['calendar']['dow'].apply(lambda x: 0 if (x < 5) else 1)
    # add holiday range size
    holidays = DataSet['calendar'][DataSet['calendar']['is_holiday'] == 1][['date']]
    holidays['hol_l0'] = utils.TagHoliday(holidays['date'].values)
    groupped = holidays.groupby(['hol_l0'])
    recs = []
    for g in groupped.groups:
        hol_days = {}
        hol_days['hol_l0'] = g
        hol_days['hol_days'] = len(groupped.get_group(g))
        recs.append(hol_days)
    tmpdf = pd.DataFrame(data=recs, index=range(len(recs)))
    holidays = holidays.merge(tmpdf, how='left', on='hol_l0')
    holidays['last_day_holiday'] = utils.IsTheLast(holidays['hol_l0'].values)
    holidays['first_day_holiday'] = utils.IsTheFirst(holidays['hol_l0'].values)
    DataSet['calendar'] = DataSet['calendar'].merge(holidays, how='left', on='date')
    DataSet['calendar'].drop(['hol_l0'], axis=1, inplace=True)
    DataSet['calendar']['hol_days'].fillna(0, inplace=True)
    DataSet['calendar']['last_day_holiday'].fillna(-1, inplace=True)
    DataSet['calendar']['first_day_holiday'].fillna(-1, inplace=True)
    # add prev/next holiday
    DataSet['calendar']['prevday'] = DataSet['calendar']['date'] - datetime.timedelta(days=1)
    DataSet['calendar']['nextday'] = DataSet['calendar']['date'] + datetime.timedelta(days=1)
    DataSet['calendar'].set_index('date', inplace=True)
    DataSet['calendar']['prev_is_holiday'] = 0
    DataSet['calendar']['prev_is_holiday'] = \
        DataSet['calendar'][DataSet['calendar']['prevday'] >= datetime.datetime(2017, 9, 5).date()]['prevday'].apply(
            lambda x: DataSet['calendar'].loc[x, 'is_holiday'])
    DataSet['calendar']['next_is_holiday'] = 0
    DataSet['calendar']['next_is_holiday'] = \
        DataSet['calendar'][DataSet['calendar']['nextday'] <= datetime.datetime(2018, 2, 5).date()]['nextday'].apply(
            lambda x: DataSet['calendar'].loc[x, 'is_holiday'])
    DataSet['calendar'].reset_index(inplace=True)
    DataSet['calendar'].drop(['prevday', 'nextday'], axis=1, inplace=True)
    DataSet['calendar']['prev_is_holiday'].fillna(0, inplace=True)
    DataSet['calendar']['next_is_holiday'].fillna(0, inplace=True)

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

raw_cols = [c for c in DataSet['train'].columns if(c.startswith('f'))]
date_cols = [c for c in DataSet['train'].columns if(c.startswith('date_'))]
total_feat_cols = raw_cols + date_cols
print(total_feat_cols)

with utils.timer('remove the unlabled'):
    for mod in ['train']:
        DataSet[mod] = DataSet[mod][DataSet[mod]['label'] != -1].reset_index(drop= True)

entire_data = pd.concat([DataSet['train'][raw_cols], DataSet['test'][raw_cols]],axis=0).reset_index(drop=True)
likely_cate_cols = []
for c in raw_cols:
    if(len(entire_data[c].value_counts()) <= 10):
        likely_cate_cols.append(c)

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'swt', utils.sum_weighted_tpr(labels, preds), True

weeks = np.max(DataSet['train']['wno']) + 1
kfold = 4
times = 1
final_cv_train_1 = np.zeros(len(DataSet['train']))
final_cv_train_2 = np.zeros(len(DataSet['train']))
final_cv_pred = np.zeros(len(DataSet['test']))

x_score = []

cv_train_1_index = DataSet['train'].index[DataSet['wno'] >= weeks - kfold]
cv_train_2_index = DataSet['train'].index[DataSet['wno'] >= weeks - kfold + 1]

for s in range(times):
    best_trees = []
    cv_train_1 = np.zeros(len(DataSet['train']))
    cv_train_2 = np.zeros(len(DataSet['train']))
    cv_pred = np.zeros(len(DataSet['test']))
    for fold in range(kfold):
        w_train_max = weeks - (kfold - fold + 1)
        train_index = DataSet['train'].index[DataSet['wno'] <= w_train_max]
        valid_1_index, valid_2_index = DataSet['train'].index[DataSet['wno'] == w_train_max + 1], DataSet['train'].index[DataSet['wno'] == w_train_max + 2]

        X_train = DataSet['train'][total_feat_cols].iloc[train_index, ].reset_index(drop= True)
        X_valid_1, X_valid_2 = DataSet['train'][total_feat_cols].iloc[valid_1_index, ].reset_index(drop= True), DataSet['train'][total_feat_cols].iloc[valid_2_index, ].reset_index(drop= True)

        y_train = DataSet['train']['label'].iloc[train_index,].reset_index(drop=True)
        y_valid_1, y_valid_2 = DataSet['train']['label'].iloc[valid_1_index,].reset_index(drop=True),DataSet['train']['label'].iloc[valid_1_index,].reset_index(drop=True)

        dtrain = lightgbm.Dataset(X_train, y_train, feature_name=total_feat_cols, categorical_feature=date_cols)
        dvalid = lightgbm.Dataset(X_valid_1, y_valid_1, reference=dtrain, feature_name=total_feat_cols, categorical_feature=date_cols)

        bst = lightgbm.train(params, dtrain, valid_sets=dvalid, feval=evalerror, verbose_eval=20,early_stopping_rounds=100)
        best_trees.append(bst.best_iteration)
        # for test
        cv_pred += bst.predict(DataSet['test'][total_feat_cols], num_iteration=bst.best_iteration)
        # for valid_1
        cv_train_1[valid_1_index] += bst.predict(X_valid_1, num_iteration=bst.best_iteration)
        # for valid_2
        cv_train_2[valid_2_index] += bst.predict(X_valid_2, num_iteration=bst.best_iteration)
        # evaluate for valid_1
        score_1 = utils.sum_weighted_tpr(y_valid_1, cv_train_1[valid_1_index])
        # evaluate for valid_2
        score_2 = utils.sum_weighted_tpr(y_valid_2, cv_train_2[valid_2_index])
        print('#%s, fold %s score_1 %.6f, score_2 %.6f' % (s, fold, score_1, score_2))
        print('valid_1: label %s, predict positives %s' % (np.sum(y_valid_1), (np.sum([1.0 if (v > 0.5) else 0 for v in cv_train_1[valid_1_index]]))))
        print('valid_2: label %s, predict positives %s' % (np.sum(y_valid_2), (np.sum([1.0 if (v > 0.5) else 0 for v in cv_train_2[valid_2_index]]))))
    cv_pred /= kfold
    final_cv_train_1 += cv_train_1
    final_cv_train_2 += cv_train_2
    final_cv_pred += cv_pred
    cv_1_score = utils.sum_weighted_tpr(DataSet['train']['label'][cv_train_1_index], cv_train_1[cv_train_1_index])
    cv_2_score = utils.sum_weighted_tpr(DataSet['train']['label'][cv_train_2_index], cv_train_2[cv_train_2_index])
    print('\n===================================')
    print('#%s, cv_1 socre %.6f, cv_2 score %.6f' % (s, cv_1_score, cv_2_score))
    print('===================================\n')

final_score_1 = utils.sum_weighted_tpr(DataSet['train']['label'][cv_train_1_index], final_cv_train_1[cv_train_1_index]/times)
final_score_2 = utils.sum_weighted_tpr(DataSet['train']['label'][cv_train_2_index], final_cv_train_2[cv_train_2_index]/times)

## output
with utils.timer("model output"):
    OutputDir = '%s/model' % config.DataRootDir
    if (os.path.exists(OutputDir) == False):
        os.makedirs(OutputDir)
    pd.DataFrame({'id': DataSet['test']['id'], 'score': final_cv_pred / (1.0 * times)}).to_csv('%s/%s_pred_avg_%.6f.csv' % (OutputDir, strategy, final_score_1), index=False)
    pd.DataFrame({'id': DataSet['train']['id'], 'score': final_cv_train_1 /(1.0 * times), 'label': DataSet['train']['label']}).to_csv('%s/%s_cv_avg_%.6f.csv' % (OutputDir, strategy, final_score_1), index=False)

end = time.time()
print('\n------------------------------------')
print('%s done, time elapsed %ss' % (strategy, int(end - start)))
print('------------------------------------\n')
