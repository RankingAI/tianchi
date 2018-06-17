import pandas as pd
import numpy as np
import datetime, sys, time, os
from contextlib import contextmanager
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import lightgbm
import utils
import config

pd.set_option('display.max_rows', None)

strategy = 'lgb_imputed'

debug = False
DataBase = './data'

impute_params = {
    'boosting': 'gbdt',
    'objective': 'regression',
    'metric': 'None',
    'max_depth': 8,
    #'num_leaves': 31,
    'num_iterations': 5000,
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    #'silent': True,
    'verbose': -10,
}

params = {
    "boosting": "gbdt",
    "objective": "binary",
    'metric': 'None',

    "num_iterations": 5000,
    "learning_rate": 0.1,  # !!!
    #'scale_pos_weight': 5,
    'max_depth': 8,
    #'min_data_in_leaf': 2000,
    'min_split_gain': 0,
    #'num_leaves': 255,

    "feature_fraction": 0.6,
    "bagging_fraction": 0.9,

    "max_bin": 255,
}

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

if __name__ == '__main__':
    ''''''
    start = time.time()
    with timer('load data'):
        ## loading data
        cal_dtypes = {
            'dow': 'uint8',
            'is_holiday': 'uint8'
        }
        dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
        DataSet = {
            'train': pd.read_csv('%s/raw/atec_anti_fraud_train.csv' % config.DataRootDir, parse_dates=['date'],
                                 date_parser=dateparse),
            'test': pd.read_csv('%s/raw/atec_anti_fraud_test_a.csv' % config.DataRootDir, parse_dates=['date'],
                                 date_parser=dateparse),
            'calendar': pd.read_csv('%s/raw/calendar.csv' % config.DataRootDir, parse_dates=['date'],
                                date_parser=dateparse, dtype=cal_dtypes, usecols=['date', 'dow', 'is_holiday'])
        }
        if(debug == True):
            tra_idx_list = [v for v in DataSet['train'].index.values if((v % 10) == 0)]
            DataSet['train'] = DataSet['train'].iloc[tra_idx_list, :].reset_index(drop=True)
            tes_idx_list = [v for v in DataSet['test'].index.values if((v % 10) == 0)]
            DataSet['test'] = DataSet['test'].iloc[tes_idx_list, :].reset_index(drop=True)
        for mod in ['calendar', 'train', 'test']:
            if (mod == 'calendar'):
                DataSet[mod]['dow'] -= 1
            DataSet[mod]['date'] = DataSet[mod]['date'].dt.date
            DataSet[mod].sort_values(by=['date'], inplace=True)

    ## add missing feature number
    with utils.timer('Add missing feat number'):
        for mod in ['train', 'test']:
            DataSet[mod]['num_missing_feat'] = DataSet[mod][config.NUMERIC_COLS + config.CATEGORICAL_COLS].isnull().sum(axis=1)

    ## groupping with the propotion of missing values
    with utils.timer('groupping'):
        raw_features = [c for c in DataSet['train'].columns if(c.startswith('f'))]
        entire_data = pd.concat([DataSet['train'][raw_features], DataSet['test'][raw_features]], axis= 0).reset_index(drop= True)
        total_size = len(entire_data)
        null_dict = {}
        for feat in raw_features:
            ratio = entire_data[feat].isnull().sum()/total_size
            null_dict[feat] = ratio
        group_null_dict = {}
        for k in null_dict.keys():
            if(null_dict[k] not in group_null_dict):
                group_null_dict[null_dict[k]] = []
            group_null_dict[null_dict[k]].append(k)
    ## check
    sorted_list = sorted(group_null_dict.items(), key= lambda x: x[0], reverse= True)
    for sl in sorted_list:
        print(sl[0], sl[1])
    print('---------------------- before imputing ------------------------------')
    print('train %s, test %s' % (len(DataSet['train']),len(DataSet['test'])))
    print(DataSet['train'][sorted_list[1][1]].isnull().sum(axis=0))
    print(DataSet['test'][sorted_list[1][1]].isnull().sum(axis=0))
    updated_data = entire_data.copy()
    ##
    for sl in sorted_list:
        g_ratio = sl[0]
        if(g_ratio == 0):
            continue
        #if(g_ratio != 0.40662401356287969):
        #    continue
        g_targets = sl[1]
        candidate_feats = [f for f in raw_features if(f not in g_targets)]
        g_start = time.time()
        for target in g_targets:
            test_index = entire_data.index[entire_data[target].isnull() == True]
            train_valid_index = entire_data.index[entire_data[target].isnull() == False]

            test = entire_data.iloc[test_index,].reset_index(drop= True)
            train_valid = entire_data.iloc[train_valid_index,].reset_index(drop= True)

            kfold = KFold(n_splits=5, random_state= 2018, shuffle=True)
            kf = kfold.split(train_valid)

            cv_train = np.zeros(len(train_valid_index))
            cv_pred = np.zeros(len(test_index))

            for i, (train_index, valid_index) in enumerate(kf):
                X_train, X_valid = train_valid[candidate_feats].iloc[train_index,], train_valid[candidate_feats].iloc[valid_index,]
                y_train, y_valid = train_valid[target].iloc[train_index, ], train_valid[target].iloc[valid_index,]

                # ##
                # y_train = np.log1p(y_train)
                # y_valid = np.log1p(y_valid)

                dtrain = lightgbm.Dataset(X_train,y_train)
                dvalid = lightgbm.Dataset(X_valid, y_valid,reference=dtrain)

                bst = lightgbm.train(impute_params, dtrain, valid_sets=dvalid, verbose_eval= None,early_stopping_rounds=100)

                ## for valid
                cv_train[valid_index] += bst.predict(X_valid)
                ## for test
                cv_pred += bst.predict(test[candidate_feats], num_iteration=bst.best_iteration)

                print('fold %s, score %.6f' % (i, mean_squared_error(y_valid, cv_train[valid_index])))
            mse = mean_squared_error(entire_data[target].iloc[train_valid_index,], cv_train)
            print('\n=================================')
            print('ratio group %.6f, target %s, mse %.6f' % (g_ratio, target, mse))
            print('==================================\n')
            cv_pred /= 5

            updated_data[target].iloc[test_index,] = list(cv_pred)

        g_end = time.time()
        print('\nGROUP %.6f DONE, TIME %s!' % (g_ratio, int(g_end - g_start)))
        ## CHECK
        #print(updated_data[g_targets].isnull().sum(axis= 0))
    train_len = len(DataSet['train'])
    DataSet['train'][raw_features] = updated_data[raw_features].iloc[:train_len,].values
    DataSet['test'][raw_features] = updated_data[raw_features].iloc[train_len:,].values
    utils.hdf_saver(DataSet['train'], '%s/raw/train.hdf' % config.DataRootDir, 'train')
    utils.hdf_saver(DataSet['test'], '%s/raw/test.hdf' % config.DataRootDir, 'test')
    print('------------------------------ after imputing -------------------------')
    print('train %s, test %s' % (len(DataSet['train']),len(DataSet['test'])))
    print(DataSet['train'][sorted_list[1][1]].isnull().sum(axis=0))
    print(DataSet['test'][sorted_list[1][1]].isnull().sum(axis=0))

    ## train with these filled values
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
            DataSet['calendar'][DataSet['calendar']['prevday'] >= datetime.datetime(2017, 9, 5).date()][
                'prevday'].apply(
                lambda x: DataSet['calendar'].loc[x, 'is_holiday'])
        DataSet['calendar']['next_is_holiday'] = 0
        DataSet['calendar']['next_is_holiday'] = \
            DataSet['calendar'][DataSet['calendar']['nextday'] <= datetime.datetime(2018, 2, 5).date()][
                'nextday'].apply(
                lambda x: DataSet['calendar'].loc[x, 'is_holiday'])
        DataSet['calendar'].reset_index(inplace=True)
        DataSet['calendar'].drop(['prevday', 'nextday'], axis=1, inplace=True)
        DataSet['calendar']['prev_is_holiday'].fillna(0, inplace=True)
        DataSet['calendar']['next_is_holiday'].fillna(0, inplace=True)
        DataSet['calendar'].drop(['hol_days'], axis=1, inplace=True)

    ## renaming
    with utils.timer('Rename columns'):
        date_col_map = dict((col, 'date_%s' % col) for col in DataSet['calendar'].columns if (col not in ['date']))
        DataSet['calendar'].rename(index=str, columns=date_col_map, inplace=True)
        cate_col_map = dict((c, 'cate_%s' % c) for c in config.CATEGORICAL_COLS)
        for mod in ['train', 'test']:
            DataSet[mod].rename(index=str, columns=cate_col_map, inplace=True)
        num_col_map = dict((c, 'num_%s' % c) for c in config.NUMERIC_COLS)
        for mod in ['train', 'test']:
            DataSet[mod].rename(index=str, columns=num_col_map, inplace=True)

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

    num_cols = [c for c in DataSet['train'].columns if (c.startswith('num_'))]
    cate_cols = [c for c in DataSet['train'].columns if (c.startswith('cate_'))]
    date_cols = [c for c in DataSet['train'].columns if (c.startswith('date_'))]
    total_feat_cols = num_cols + cate_cols + date_cols

    with utils.timer('remove the unlabled'):
        for mod in ['train']:
            DataSet[mod] = DataSet[mod][DataSet[mod]['label'] != -1].reset_index(drop=True)

    def evalerror(preds, dtrain):
        labels = dtrain.get_label()
        return 'swt', utils.sum_weighted_tpr(labels, preds), True

    kfold = 5
    times = 8
    final_cv_train = np.zeros(len(DataSet['train']))
    final_cv_pred = np.zeros(len(DataSet['test']))
    cv_precision = np.zeros((times, kfold), dtype=np.float)
    skf = StratifiedKFold(n_splits=kfold, random_state=None, shuffle=True)

    x_score = []
    for s in range(times):
        cv_train = np.zeros(len(DataSet['train']))
        cv_pred = np.zeros(len(DataSet['test']))

        params['seed'] = 2018

        kf = skf.split(DataSet['train'][total_feat_cols], DataSet['train']['label'])

        best_trees = []
        fold_scores = []

        with utils.timer('train/inference'):
            for fold, (train_index, valid_index) in enumerate(kf):
                X_train, X_valid = DataSet['train'][total_feat_cols].iloc[train_index,], \
                                   DataSet['train'][total_feat_cols].iloc[valid_index,]
                y_train, y_valid = DataSet['train']['label'][train_index], DataSet['train']['label'][valid_index]
                dtrain = lightgbm.Dataset(X_train,
                                          y_train)  # , feature_name= total_feat_cols, categorical_feature= cate_cols + date_cols)
                dvalid = lightgbm.Dataset(X_valid, y_valid,
                                          reference=dtrain)  # , feature_name= total_feat_cols, categorical_feature= cate_cols + date_cols)
                bst = lightgbm.train(params, dtrain, valid_sets=dvalid, feval=evalerror, verbose_eval=20,
                                     early_stopping_rounds=100)
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
        t_score = utils.sum_weighted_tpr(DataSet['train']['label'], cv_train)
        c_score = utils.sum_weighted_tpr(DataSet['train']['label'], final_cv_train / (s + 1.0))
        print('#%s CV score %.6f' % (s, t_score))
        print('#%s Current score %.6f' % (s, c_score))
        print(fold_scores)
        print(best_trees, np.mean(best_trees))
        print('===================\n')

        x_score.append(t_score)

    print(x_score)

    final_score = utils.sum_weighted_tpr(DataSet['train']['label'], final_cv_train / times)
    ## output
    with utils.timer("model output"):
        OutputDir = '%s/model' % config.DataRootDir
        if (os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)
        pd.DataFrame({'id': DataSet['test']['id'], 'score': final_cv_pred / (1.0 * times)}).to_csv(
            '%s/%s_pred_avg_%.6f.csv' % (OutputDir, strategy, final_score), index=False)
        pd.DataFrame({'id': DataSet['train']['id'], 'score': final_cv_train / (1.0 * times),
                      'label': DataSet['train']['label']}).to_csv(
            '%s/%s_cv_avg_%.6f.csv' % (OutputDir, strategy, final_score), index=False)

    end = time.time()
    print('\n------------------------------------')
    print('%s done, time elapsed %ss' % (strategy, int(end - start)))
    print('------------------------------------\n')
