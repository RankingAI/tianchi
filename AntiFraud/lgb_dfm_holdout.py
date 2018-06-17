import sys, os, time, datetime
import numpy as np
import pandas as pd
import config
import utils
from sklearn.model_selection import StratifiedKFold
import lightgbm
from sklearn.linear_model import LogisticRegression

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

strategy = 'lgb_dfm_holdout'
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
#drop_cols = [c for c in date_cols if(c not in ['date_dow', 'date_is_holiday'])]
#date_cols = [c for c in date_cols if(c not in drop_cols)]
#total_feat_cols = [c for c in total_feat_cols if(c not in drop_cols)]
print(total_feat_cols)

with utils.timer('remove the unlabled'):
    for mod in ['train']:
        DataSet[mod] = DataSet[mod][DataSet[mod]['label'] != -1].reset_index(drop= True)

# entire_data = pd.concat([DataSet['train'][raw_cols], DataSet['test'][raw_cols]],axis=0).reset_index(drop=True)
# likely_cate_cols = []
# for c in raw_cols:
#     if(len(entire_data[c].value_counts()) <= 10):
#         likely_cate_cols.append(c)

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'swt', utils.sum_weighted_tpr(labels, preds), True

weeks = np.max(DataSet['train']['wno']) + 1
kfold = 5
times = 4
final_cv_train = np.zeros(len(DataSet['train']))
final_cv_pred = np.zeros(len(DataSet['test']))
skf = StratifiedKFold(n_splits= kfold, random_state= None, shuffle= False)

## validate with the last week
valid_index = DataSet['train'].index[DataSet['train']['wno'] == weeks - 1]

x_score = []
for s in range(times):
    params['seed'] = s
    with utils.timer('train/inference'):

        # ## single model for lgb
        # lgb_cv_train_avg = np.zeros(len(DataSet['train']))
        # lgb_cv_train = np.zeros((len(DataSet['train']), weeks - 1))
        # lgb_cv_pred = np.zeros((len(DataSet['test']), weeks - 1))
        # for w in range(weeks - 1):
        #     train_index = DataSet['train'].index[DataSet['train']['wno'] == w]
        #
        #     X_train, X_valid = DataSet['train'][total_feat_cols].iloc[train_index,].reset_index(drop= True), DataSet['train'][total_feat_cols].iloc[valid_index,].reset_index(drop= True)
        #     y_train, y_valid = DataSet['train']['label'].iloc[train_index,].reset_index(drop= True), DataSet['train']['label'].iloc[valid_index,].reset_index(drop= True)
        #
        #     dtrain = lightgbm.Dataset(X_train, y_train, feature_name= total_feat_cols, categorical_feature= date_cols)
        #     dvalid = lightgbm.Dataset(X_valid, y_valid, reference= dtrain, feature_name= total_feat_cols, categorical_feature= date_cols)
        #     # train
        #     bst = lightgbm.train(params, dtrain, valid_sets=dvalid, feval=evalerror, verbose_eval= 20,early_stopping_rounds= 100)
        #     # predict on test
        #     lgb_cv_pred[:,w] += bst.predict(DataSet['test'][total_feat_cols], num_iteration=bst.best_iteration)
        #     # predict on valid
        #     tmp_pred = bst.predict(X_valid, num_iteration= bst.best_iteration)
        #     lgb_cv_train[valid_index, w] += tmp_pred
        #     lgb_cv_train_avg[valid_index] += tmp_pred
        #     # evaluate for valid
        #     score = utils.sum_weighted_tpr(y_valid, lgb_cv_train[valid_index, w])
        #     print('\n--------------------------------')
        #     print('#%s, week %s score %.6f' % (s, w, score))
        #     print('valid: label %s, predict positives %s' % (np.sum(y_valid), (np.sum([1.0 if(v > 0.5) else 0 for v in lgb_cv_train[valid_index, w]]))))
        #     print('--------------------------------\n')
        #
        # lgb_cv_score_avg = utils.sum_weighted_tpr(DataSet['train']['label'][valid_index], lgb_cv_train_avg[valid_index]/(weeks - 1))

        ## single model for dfm
        dfm_cv_train_avg = np.zeros(len(DataSet['train']))
        dfm_cv_train = np.zeros((len(DataSet['train']), weeks - 1))
        dfm_cv_pred = np.zeros((len(DataSet['test']), weeks - 1))
        for w in range(weeks - 1):
            train_index = DataSet['train'].index[DataSet['train']['wno'] == w]

            X_train, X_valid = DataSet['train'][total_feat_cols].iloc[train_index,].reset_index(drop= True), DataSet['train'][total_feat_cols].iloc[valid_index,].reset_index(drop= True)
            y_train, y_valid = DataSet['train']['label'].iloc[train_index,].reset_index(drop= True), DataSet['train']['label'].iloc[valid_index,].reset_index(drop= True)




