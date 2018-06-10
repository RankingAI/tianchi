import sys, os, time, datetime
import numpy as np
import pandas as pd
import config
import utils
from sklearn.model_selection import StratifiedKFold
import lightgbm

sys.path.append("..")
pd.set_option('display.max_rows', None)
# parameters
params = {
    "boosting": "gbdt",
    "objective": "binary",
    "lambda_l2": 2,  # !!!
    #'metric': 'auc',

    "num_iterations": 200,
    "learning_rate": 0.05,  # !!!
    "max_depth": 8,  # !!!
    'scale_pos_weight': 9,
    #'min_data_in_leaf': 50,
    #'num_leaves': 256,
    'cat_smooth': 80,
    'cat_l2': 20,
    #'max_cat_threshold': 64,

    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    #"bagging_freq": 20,
    #"min_hessian": 0.001,

    "max_bin": 63,
}
strategy = 'lgb'

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

## renaming
with utils.timer('Rename columns'):
    date_col_map = dict((col, 'date_%s' % col) for col in DataSet['calendar'].columns if (col not in ['date']))
    DataSet['calendar'].rename(index=str, columns= date_col_map, inplace=True)
    cate_col_map = dict((c, 'cate_%s' % c) for c in config.CATEGORICAL_COLS)
    for mod in ['train', 'test']:
        DataSet[mod].rename(index= str, columns= cate_col_map, inplace= True)
    num_col_map = dict((c , 'num_%s' % c) for c in config.NUMERIC_COLS)
    for mod in ['train', 'test']:
        DataSet[mod].rename(index= str, columns= num_col_map, inplace= True)

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

num_cols = [c for c in DataSet['train'].columns if(c.startswith('num_'))]
cate_cols = [c for c in DataSet['train'].columns if(c.startswith('cate_'))]
date_cols = [c for c in DataSet['train'].columns if(c.startswith('date_'))]
total_feat_cols = num_cols + cate_cols + date_cols
# ## fill the missing
# with utils.timer('Fill the missing'):
#     for mod in ['train', 'test']:
#         for c in cate_cols:
#             DataSet[mod][c].fillna(-1, inplace= True)
#         for c in num_cols:
#             DataSet[mod][c].fillna(0, inplace= True)

with utils.timer('remove the unlabled'):
    for mod in ['train']:
        DataSet[mod] = DataSet[mod][DataSet[mod]['label'] != -1].reset_index(drop= True)

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'swt', utils.sum_weighted_tpr(labels, preds), True

kfold = 5
times = 8
final_cv_train = np.zeros(len(DataSet['train']))
final_cv_pred = np.zeros(len(DataSet['test']))
cv_precision = np.zeros((times, kfold), dtype= np.float)
skf = StratifiedKFold(n_splits= kfold, random_state= 2018, shuffle= True)

x_score = []
for s in range(times):
    cv_train = np.zeros(len(DataSet['train']))
    cv_pred = np.zeros(len(DataSet['test']))

    params['seed'] = s

    kf = skf.split(DataSet['train'][total_feat_cols], DataSet['train']['label'])

    best_trees = []
    fold_scores = []

    with utils.timer('train/inference'):
        for fold, (train_index, valid_index) in enumerate(kf):
            X_train, X_valid = DataSet['train'][total_feat_cols].iloc[train_index,], DataSet['train'][total_feat_cols].iloc[valid_index,]
            y_train, y_valid = DataSet['train']['label'][train_index], DataSet['train']['label'][valid_index]
            dtrain = lightgbm.Dataset(X_train, y_train, feature_name= total_feat_cols, categorical_feature= cate_cols + date_cols)
            dvalid = lightgbm.Dataset(X_valid, y_valid, feature_name= total_feat_cols, categorical_feature= cate_cols + date_cols, reference= dtrain)
            bst = lightgbm.train(params, dtrain, valid_sets=dvalid, feval=evalerror, verbose_eval=10,early_stopping_rounds= 20)
            best_trees.append(bst.best_iteration)
            cv_pred += bst.predict(DataSet['test'][total_feat_cols], num_iteration=bst.best_iteration)
            cv_train[valid_index] += bst.predict(X_valid)

            score = utils.sum_weighted_tpr(y_valid, cv_train[valid_index])
            fold_scores.append(score)
            print('#%s fold %s %.6f' % (s, fold, score))

    cv_pred /= kfold
    final_cv_train += cv_train
    final_cv_pred += cv_pred

    print('\n===================')
    print('#%s CV score %.6f' % (s, utils.sum_weighted_tpr(DataSet['train']['label'], cv_train)))
    print('#%s Current score %.6f' % (s, utils.sum_weighted_tpr(DataSet['train']['label'], final_cv_train/(s + 1.0))))
    print(fold_scores)
    print(best_trees, np.mean(best_trees))
    print('===================\n')

    x_score.append(utils.sum_weighted_tpr(DataSet['train']['label'], cv_train))

print(x_score)

## output
with utils.timer("model output"):
    OutputDir = '%s/model' % config.DataRootDir
    if (os.path.exists(OutputDir) == False):
        os.makedirs(OutputDir)
    pd.DataFrame({'id': DataSet['test']['id'], 'score': final_cv_pred / (1.0 * times)}).to_csv('%s/%s_pred_avg.csv' % (OutputDir, strategy), index=False)
    pd.DataFrame({'id': DataSet['train']['id'], 'score': final_cv_train /(1.0 * times), 'label': DataSet['train']['label']}).to_csv('%s/%s_cv_avg.csv' % (OutputDir, strategy), index=False)

end = time.time()
print('\n------------------------------------')
print('%s done, time elapsed %ss' % (strategy, int(end - start)))
print('------------------------------------\n')
