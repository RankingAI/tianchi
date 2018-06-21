########################################################################
# Self-Training versioned Semi-supervised Learning with Segmented data #
########################################################################
import sys, os, time, datetime
import numpy as np
import pandas as pd
import config
import utils
from sklearn.model_selection import StratifiedKFold
import lightgbm
from sklearn.metrics import roc_auc_score

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
    'num_leaves': 255,
    'max_depth': 8,  # !!!
    'scale_pos_weight': 1,
    'verbose': -10,

    'feature_fraction': 0.6,
    'bagging_fraction': 0.9,

    'max_bin': 255,
}
strategy = 'lgb_sss'
debug = True
sl_sampling_rate = 0.1

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
    DataSet['calendar']['prev_is_holiday'] = DataSet['calendar'][DataSet['calendar']['prevday'] >= datetime.datetime(2017, 9, 5).date()]['prevday'].apply(lambda x: DataSet['calendar'].loc[x, 'is_holiday'])
    DataSet['calendar']['next_is_holiday'] = 0
    DataSet['calendar']['next_is_holiday'] = DataSet['calendar'][DataSet['calendar']['nextday'] <= datetime.datetime(2018, 2, 5).date()]['nextday'].apply(lambda x: DataSet['calendar'].loc[x, 'is_holiday'])
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

## treat the whole raw features as the numeric ones
raw_cols = [c for c in DataSet['train'].columns if(c.startswith('f'))]
date_cols = [c for c in DataSet['train'].columns if(c.startswith('date_'))]
#drop_cols = [c for c in date_cols if(c in ['date_hol_days', 'date_dow'])]
#date_cols = [c for c in date_cols if(c not in drop_cols)]
total_feat_cols = raw_cols# + date_cols
#print(total_feat_cols)

## removing the unlabeled data by now
with utils.timer('remove the unlabled'):
    for mod in ['train']:
        DataSet[mod] = DataSet[mod][DataSet[mod]['label'] != -1].reset_index(drop= True)

## likely categorical features
entire_data = pd.concat([DataSet['train'][raw_cols], DataSet['test'][raw_cols]],axis=0).reset_index(drop=True)
likely_cate_cols = []
for c in raw_cols:
    if(len(entire_data[c].value_counts()) <= 10):
        likely_cate_cols.append(c)

def evalauc(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', roc_auc_score(labels, [1 if(v > 0.5) else 0 for v in preds]), True

def evaltpr(preds, dtrain):
    labels = dtrain.get_label()
    return 'tpr', utils.sum_weighted_tpr(labels, preds), True

def public_train(data, weeks, kfold):
    times = 1
    final_cv_train = np.zeros(len(data['train']))
    final_cv_pred = np.zeros(len(data['test']))

    fold_tpr_scores = np.zeros((times, weeks, kfold))
    fold_auc_scores = np.zeros((times, weeks, kfold))

    week_cv_auc_scores = np.zeros((times, weeks))
    week_cv_tpr_scores = np.zeros((times, weeks))

    week_conf_intval = np.zeros((times, threshold_week, 6))

    t_cv_tpr_scores = np.zeros(times)
    t_agg_cv_tpr_scores = np.zeros(times)
    t_cv_auc_scores = np.zeros(times)
    t_agg_cv_auc_scores = np.zeros(times)

    skf = StratifiedKFold(n_splits= kfold, random_state=None, shuffle=False)

    start = time.time()

    for s in range(times):
        params['seed'] = s

        cv_train = np.zeros(len(data['train']))
        cv_pred = np.zeros(len(data['test']))

        s_start = time.time()

        for w in range(weeks):

            best_trees = []

            week_index = data['train'].index[data['train']['wno'] == w]
            week_data = data['train'].iloc[week_index,].reset_index(drop= True)

            cv_train_week = np.zeros(len(week_index))
            cv_pred_week = np.zeros(len(data['test']))

            kf = skf.split(week_data[total_feat_cols], week_data['label'])

            w_start = time.time()
            for fold, (train_index, valid_index) in enumerate(kf):
                f_start = time.time()

                X_train, X_valid = week_data[total_feat_cols].iloc[train_index,].reset_index(drop=True), week_data[total_feat_cols].iloc[valid_index,].reset_index(drop=True)
                y_train, y_valid = week_data['label'].iloc[train_index,].reset_index(drop=True), week_data['label'].iloc[valid_index,].reset_index(drop=True)

                dtrain = lightgbm.Dataset(X_train, y_train, feature_name=total_feat_cols)#, categorical_feature= date_cols)#likely_cate_cols)
                dvalid = lightgbm.Dataset(X_valid, y_valid, reference=dtrain, feature_name=total_feat_cols)#, categorical_feature= date_cols)#likely_cate_cols)

                bst = lightgbm.train(params, dtrain, num_iterations, valid_sets=dvalid, feval= evaltpr, verbose_eval= 50,early_stopping_rounds= 100)

                best_trees.append(bst.best_iteration)
                ## predict with single-week model
                cv_pred_week += bst.predict(data['test'][total_feat_cols],num_iteration=bst.best_iteration)  # * week_weight
                cv_train_week[valid_index] += bst.predict(X_valid, num_iteration=bst.best_iteration)  # * week_weight

                label_positives = np.sum(y_valid)
                pred_positives = (np.sum([1.0 if (v > 0.5) else 0 for v in cv_train_week[valid_index]]))

                tpr_score = utils.sum_weighted_tpr(y_valid, cv_train_week[valid_index])
                auc_score = roc_auc_score(y_valid, [1 if(v > 0.5) else 0 for v in cv_train_week[valid_index]])

                fold_tpr_scores[s][w][fold] = tpr_score
                fold_auc_scores[s][w][fold] = auc_score

                f_end = time.time()

                print('\n---------------------------------------------------')
                print('#%s: week %s, fold %s, score %.6f/%.6f, positives %s/%s' % (s, w, fold,
                                                                                   tpr_score,
                                                                                   auc_score,
                                                                                   label_positives,
                                                                                   pred_positives))
                print('time elapsed %s' % int(f_end - f_start))
                print('---------------------------------------------------\n')

            w_end = time.time()
            cv_pred_week /= kfold
            ## aggregate cv_pred
            cv_pred += list(cv_pred_week)
            cv_train[week_index] = list(cv_train_week)

            week_cv_tpr_score = utils.sum_weighted_tpr(week_data['label'], cv_train_week)
            week_cv_auc_score = roc_auc_score(week_data['label'], [1 if(v > 0.5) else 0 for v in cv_train_week])

            sorted_pred = np.sort(cv_pred_week, axis= None)
            week_conf_intval[s][w][0] = len(sorted_pred[np.where(sorted_pred > 0.8)])/len(sorted_pred)
            week_conf_intval[s][w][1] = len(sorted_pred[np.where(sorted_pred > 0.7)])/len(sorted_pred)
            week_conf_intval[s][w][2] = len(sorted_pred[np.where(sorted_pred > 0.6)])/len(sorted_pred)

            week_conf_intval[s][w][3] = len(sorted_pred[np.where(sorted_pred < 0.1)])/len(sorted_pred)
            week_conf_intval[s][w][4] = len(sorted_pred[np.where(sorted_pred < 0.05)])/len(sorted_pred)
            week_conf_intval[s][w][5] = len(sorted_pred[np.where(sorted_pred < 0.01)])/len(sorted_pred)

            label_positives = np.sum(week_data['label'])
            pred_positives = np.sum([1.0 if (v > 0.5) else 0 for v in cv_train_week])

            week_cv_tpr_scores[s][w] = week_cv_tpr_score
            week_cv_auc_scores[s][w] = week_cv_auc_score

            print(week_conf_intval[s][w])

            print('\n------------------------------------------')
            print('#%s, week %s, cv score %.6f/%.6f, positives %s/%s' % (s, w,
                                                                                               week_cv_tpr_score,
                                                                                               week_cv_auc_score,
                                                                                               label_positives,
                                                                                               pred_positives)
                  )
            print(best_trees, np.mean(best_trees))
            print('time elapsed %s' % (int(w_end - w_start)))
            print('------------------------------------------\n')
        s_end = time.time()
        ## average cv_pred by weeks
        cv_pred /= weeks

        final_cv_train += cv_train
        final_cv_pred += cv_pred

        c_cv_tpr_score = utils.sum_weighted_tpr(data['train']['label'], cv_train)
        a_cv_tpr_score = utils.sum_weighted_tpr(data['train']['label'], final_cv_train / (s + 1.0))

        c_cv_auc_score = roc_auc_score(data['train']['label'], [1 if(v > 0.5) else 0 for v in cv_train])
        a_cv_auc_score = roc_auc_score(data['train']['label'], [1 if(v > 0.5) else 0 for v in (final_cv_train/(s + 1.0)).tolist()])

        t_cv_tpr_scores[s] = c_cv_tpr_score
        t_agg_cv_tpr_scores[s] = a_cv_tpr_score

        t_cv_auc_scores[s] = c_cv_auc_score
        t_agg_cv_auc_scores[s] = a_cv_auc_score

        print('\n====================================================')
        print('#%s: current cv score %.6f/%.6f, aggregated cv score %.6f/%.6f' % (s, c_cv_tpr_score,
                                                                                  c_cv_auc_score,
                                                                                  a_cv_tpr_score,
                                                                                  a_cv_auc_score))
        print('time elapsed %s' % (s_end - s_start))
        print('====================================================\n')

    final_score = utils.sum_weighted_tpr(data['train']['label'], final_cv_train / times)

    print(t_cv_tpr_scores)
    print(t_agg_cv_tpr_scores)

    print(week_conf_intval)

    ## output
    with utils.timer("model output"):
        OutputDir = '%s/model' % config.DataRootDir
        if (os.path.exists(OutputDir) == False):
            os.makedirs(OutputDir)
        pd.DataFrame({'id': data['train']['id'], 'score': final_cv_train / (1.0 * times),'label': data['train']['label']}).to_csv('%s/%s_cv_train_%.6f.csv' % (OutputDir, strategy, final_score), index=False)
        pd.DataFrame({'id': data['test']['id'], 'score': final_cv_pred / (1.0 * times)}).to_csv('%s/%s_cv_pred_%.6f.csv' % (OutputDir, strategy, final_score), index=False)
    end = time.time()
    print('\n------------------------------------')
    print('%s done, time elapsed %ss' % (strategy, int(end - start)))
    print('------------------------------------\n')

def train_with_cv(train_data, test_data, kfold, skf, cv_params, s, w):
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

        dtrain = lightgbm.Dataset(X_train, y_train,feature_name=total_feat_cols)  # , categorical_feature= date_cols)#likely_cate_cols)
        dvalid = lightgbm.Dataset(X_valid, y_valid, reference=dtrain,feature_name=total_feat_cols)  # , categorical_feature= date_cols)#likely_cate_cols)

        bst = lightgbm.train(cv_params, dtrain, num_iterations, valid_sets=dvalid, feval=evaltpr, verbose_eval=50,early_stopping_rounds=100)
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

def proba2label(data):
    return [1 if(v > 0.5) else 0 for v in data]

def local_train_evaluate(data, weeks, kfold):
    ''''''
    times = 1
    final_cv_train = np.zeros(len(data))
    final_cv_pred = np.zeros(len(data))

    ## train with the first five weeks, while evaluating on the last four weeks in train data set
    threshold_week = int(weeks/2) + 1
    train_valid_index = data.index[data['wno'] < threshold_week]
    test_index = data.index[data['wno'] >= threshold_week]
    test_data = data.iloc[test_index,].reset_index(drop= True)

    ## for summary
    fold_tpr_scores = np.zeros((times, threshold_week, kfold))
    fold_auc_scores = np.zeros((times, threshold_week, kfold))

    week_cv_auc_scores = np.zeros((times, threshold_week))
    week_cv_tpr_scores = np.zeros((times, threshold_week))
    week_eval_auc_scores = np.zeros((times, threshold_week))
    week_eval_tpr_scores = np.zeros((times, threshold_week))

    week_conf_intval = np.zeros((times, threshold_week, 6))

    t_cv_tpr_scores = np.zeros(times)
    t_agg_cv_tpr_scores = np.zeros(times)
    t_eval_tpr_scores = np.zeros(times)
    t_agg_eval_tpr_scores = np.zeros(times)
    t_cv_auc_scores = np.zeros(times)
    t_agg_cv_auc_scores = np.zeros(times)
    t_eval_auc_scores = np.zeros(times)
    t_agg_eval_auc_scores = np.zeros(times)

    ##
    skf = StratifiedKFold(n_splits= kfold, random_state=None, shuffle=False)

    for s in range(times):
        params['seed'] = s

        cv_train = np.zeros(len(data))
        cv_pred = np.zeros(len(data))

        s_start = time.time()

        for w in range(threshold_week):

            w_start = time.time()

            best_trees = []
            ## prepare labeled data for CV, L
            week_index = data.index[data['wno'] == w]
            week_data = data.iloc[week_index,].reset_index(drop= True)

            ## pre-train with L and sampling from U with confidence interval, U' is the output of this step
            with utils.timer('Pre-train and PL sampling'):
                # pre-train with CV
                pre_cv_train_week, pre_cv_pred_week, _, _ = train_with_cv(week_data, test_data, kfold, skf, params, s, w)
                pre_cv_tpr_scores = utils.sum_weighted_tpr(week_data['label'], pre_cv_train_week)
                pre_eval_tpr_scores = utils.sum_weighted_tpr(test_data['label'], pre_cv_pred_week)
                # sampling for pseudo label, U' is the output
                ## option 1
                # sorted_preds = np.sort(pre_cv_pred_week, axis=None)
                # down_thre, up_thre = sorted_preds[:int(0.05 * len(sorted_preds))][-1], sorted_preds[-int(0.05 * len(sorted_preds)):][0]
                # pl_sample_index = np.where(((pre_cv_pred_week >= up_thre) | (pre_cv_pred_week <= down_thre)))[0]
                # remained_index = np.where((pre_cv_pred_week < up_thre) & (pre_cv_pred_week > down_thre))[0]
                ## option 2
                pl_sample_index = test_data.index[:int(0.1 * len(test_data))]
                remained_index = test_data.index[int(0.1 * len(test_data)):]
                pl_data = test_data[week_data.columns].iloc[pl_sample_index,].reset_index(drop= True)

            ## re-train with L and U'
            with utils.timer('Post-train'):
                post_week_data = pd.concat([week_data, pl_data], axis= 0, ignore_index= True)
                post_test_data = test_data.iloc[remained_index,].reset_index(drop= True)
                post_cv_train_week, post_cv_pred_week, _, _ = train_with_cv(post_week_data, post_test_data, kfold, skf, params, s, w)
                cv_train_week = post_cv_train_week[:len(week_index)]
                cv_pred_week = np.hstack([post_cv_train_week[len(week_index):], post_cv_pred_week])

            week_cv_tpr_scores[s][w] = utils.sum_weighted_tpr(week_data['label'], cv_train_week)
            week_cv_auc_scores[s][w] = roc_auc_score(week_data['label'], proba2label(cv_train_week))

            assert (len(test_data) == len(cv_pred_week))
            week_eval_tpr_scores[s][w] = utils.sum_weighted_tpr(test_data['label'], cv_pred_week)
            week_eval_auc_scores[s][w] = roc_auc_score(test_data['label'], proba2label(cv_pred_week))

            print('\n========================================')
            print('cv tpr scores for pre-train:')
            print(pre_cv_tpr_scores)
            print('cv trp score for post-train:')
            print(week_cv_tpr_scores[s][w])
            print('------------------------------------------')
            print('eval tpr scores for pre-train:')
            print(pre_eval_tpr_scores)
            print('eval tpr score for post-train:')
            print(week_eval_tpr_scores[s][w])
            print('========================================\n')

            w_end = time.time()
            ## aggregate cv_pred
            cv_pred[test_index] += list(cv_pred_week)
            cv_train[week_index] = list(cv_train_week)

            label_positives = np.sum(week_data['label'])
            pred_positives = np.sum(proba2label(cv_train_week))

            # sorted_pred = np.sort(cv_pred_week, axis= None)
            # week_conf_intval[s][w][0] = len(sorted_pred[np.where(sorted_pred > 0.8)])/len(sorted_pred)
            # week_conf_intval[s][w][1] = len(sorted_pred[np.where(sorted_pred > 0.7)])/len(sorted_pred)
            # week_conf_intval[s][w][2] = len(sorted_pred[np.where(sorted_pred > 0.6)])/len(sorted_pred)
            #
            # week_conf_intval[s][w][3] = len(sorted_pred[np.where(sorted_pred < 0.1)])/len(sorted_pred)
            # week_conf_intval[s][w][4] = len(sorted_pred[np.where(sorted_pred < 0.05)])/len(sorted_pred)
            # week_conf_intval[s][w][5] = len(sorted_pred[np.where(sorted_pred < 0.01)])/len(sorted_pred)
            #
            # print(week_conf_intval[s][w])

            print('\n------------------------------------------')
            print('#%s, week %s, cv score %.6f/%.6f, eval score %.6f/%.6f, positives %s/%s' % (s, w,
                                                                                               week_cv_tpr_scores[s][w],week_cv_auc_scores[s][w],
                                                                                               week_eval_tpr_scores[s][w],week_eval_auc_scores[s][w],
                                                                                               label_positives,pred_positives)
                  )
            print('time elapsed %s' % (int(w_end - w_start)))
            print('------------------------------------------\n')

        s_end = time.time()
        ## average cv_pred by weeks
        cv_pred /= threshold_week

        final_cv_train += cv_train
        final_cv_pred += cv_pred

        c_cv_tpr_score = utils.sum_weighted_tpr(data['label'].iloc[train_valid_index,], cv_train[train_valid_index])
        a_cv_tpr_score = utils.sum_weighted_tpr(data['label'].iloc[train_valid_index,], final_cv_train[train_valid_index] / (s + 1.0))
        c_eval_tpr_score = utils.sum_weighted_tpr(data['label'].iloc[test_index,], cv_pred[test_index])
        a_eval_tpr_score = utils.sum_weighted_tpr(data['label'].iloc[test_index,], final_cv_pred[test_index]/(s + 1.0))

        c_cv_auc_score = roc_auc_score(data['label'].iloc[train_valid_index,], [1 if(v > 0.5) else 0 for v in cv_train[train_valid_index]])
        a_cv_auc_score = roc_auc_score(data['label'].iloc[train_valid_index,], [1 if(v > 0.5) else 0 for v in (final_cv_train[train_valid_index]/(s + 1.0)).tolist()])
        c_eval_auc_score = roc_auc_score(data['label'].iloc[test_index,], [1 if(v > 0.5) else 0 for v in cv_pred[test_index]])
        a_eval_auc_score = roc_auc_score(data['label'].iloc[test_index,], [1 if(v > 0.5) else 0 for v in (final_cv_pred[test_index]/(s + 1.0)).tolist()])

        t_cv_tpr_scores[s] = c_cv_tpr_score
        t_agg_cv_tpr_scores[s] = a_cv_tpr_score
        t_eval_tpr_scores[s] = c_eval_tpr_score
        t_agg_eval_tpr_scores[s] = a_eval_tpr_score

        t_cv_auc_scores[s] = c_cv_auc_score
        t_agg_cv_auc_scores[s] = a_cv_auc_score
        t_eval_auc_scores[s] = c_eval_auc_score
        t_agg_eval_auc_scores[s] = a_eval_auc_score

        print('\n====================================================')
        print('#%s: current cv score %.6f/%.6f, aggregated cv score %.6f/%.6f' % (s, c_cv_tpr_score,
                                                                                  c_cv_auc_score,
                                                                                  a_cv_tpr_score,
                                                                                  a_cv_auc_score))
        print('#%s: current eval score %.6f/%.6f, aggregated eval score %.6f/%.6f' % (s, c_eval_tpr_score,
                                                                                      c_eval_auc_score,
                                                                                      a_eval_tpr_score,
                                                                                      a_eval_auc_score))
        print('time elapsed %s' % (s_end - s_start))
        print('====================================================\n')

    ## summary
    print('\nfold tpr scores')
    print(fold_tpr_scores)
    print('\nfold auc scores')
    print(fold_auc_scores)
    print('\n----------------------')

    print('\nweek cv tpr scores')
    print(week_cv_tpr_scores)
    print('\nweek cv auc scores')
    print(week_cv_auc_scores)
    print('\nweek eval tpr scores')
    print(week_eval_tpr_scores)
    print('\nweek eval auc scores')
    print(week_eval_auc_scores)
    print('\n----------------------')
    print('\ncurrent cv tpr scores')
    print(t_cv_tpr_scores)
    print('\ncurrent cv auc scores')
    print(t_cv_auc_scores)
    print('\naggregated cv tpr scores')
    print(t_agg_cv_tpr_scores)
    print('\naggregated cv auc scores')
    print(t_agg_cv_auc_scores)
    print('\n----------------------')
    print('\ncurrent eval tpr scores')
    print(t_eval_tpr_scores)
    print('\ncurrent eval auc scores')
    print(t_eval_auc_scores)
    print('\naggregated eval tpr scores')
    print(t_agg_eval_tpr_scores)
    print('\naggregated eval auc scores')
    print(t_agg_eval_auc_scores)

    print('\n----------------------')
    print(week_conf_intval)

## total weeks
weeks = np.max(DataSet['train']['wno']) + 1
print('Weeks %s' % weeks)

## train and evaluate with local mode
local_train_evaluate(DataSet['train'], weeks, 4)
#public_train(DataSet, weeks, 4)
