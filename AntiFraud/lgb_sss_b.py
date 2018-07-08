########################################################################
# Self-Training versioned Semi-supervised Learning with Segmented data #
########################################################################
import sys, os, time, datetime, psutil
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
process = psutil.Process(os.getpid())
def _print_memory_usage():
    ''''''
    print('\n---- Current memory usage %sM ----\n' % int(process.memory_info().rss/(1024*1024)))

num_iterations =  5000
params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    #'metric': 'binary_logloss',
    'metric': 'None',

    #'num_iterations': 5000,
    'learning_rate': 0.01,  # !!!
    'num_leaves': 127,
    'max_depth': 8,  # !!!
    'scale_pos_weight': 1,
    'verbose': -10,

    'feature_fraction': 0.6,
    'bagging_fraction': 0.9,

    'num_threads': -1,

    'max_bin': 255,
}
strategy = 'lgb_sss'
debug = False
pl_sampling_rate = 0.1
pl_sample_weight = 1.0
pl_sampling_times = 1
train_times = 32
unlabeled_weight = 0.95

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

##
entire_data = pd.concat([DataSet['train'][raw_cols], DataSet['test'][raw_cols]], axis=0).reset_index(drop=True)
total_size = len(entire_data)
null_dict = {}
for feat in raw_cols:
    ratio = entire_data[feat].isnull().sum() / total_size
    null_dict[feat] = ratio
del entire_data
gc.collect()
group_null_dict = {}
for k in null_dict.keys():
    if (null_dict[k] not in group_null_dict):
        group_null_dict[null_dict[k]] = []
    group_null_dict[null_dict[k]].append(k)
sorted_group_null = sorted(group_null_dict.items(), key=lambda x: x[0], reverse=True)
g_idx = 0
for sgn in sorted_group_null:
    DataSet['train']['group_null_%s' % g_idx] = DataSet['train'][sgn[1]].isnull().sum(axis= 1)
    DataSet['test']['group_null_%s' % g_idx] = DataSet['test'][sgn[1]].isnull().sum(axis= 1)
    g_idx += 1

group_null_cols = [c for c in DataSet['train'].columns if(c.startswith('group_null'))]
total_feat_cols = raw_cols + ['num_missing_feat'] + group_null_cols

print(DataSet['train'][group_null_cols].head())

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
    #pl_sample_index = np.where((proba_array >= up_thre))[0]
    #remained_index = np.where((proba_array < up_thre))[0]

    print(down_thre, up_thre)

    return pl_sample_index, remained_index

def public_train(data, weeks, kfold):
    data['test']['label'] = 0

    ## for final
    pre_final_cv_train = np.zeros(len(data['train']))
    pre_final_cv_pred = np.zeros(len(data['test']))
    final_cv_train = np.zeros(len(data['train']))
    final_cv_pred = np.zeros(len(data['test']))

    ## for week
    pre_week_cv_tpr_scores = np.zeros((train_times, weeks))
    week_cv_tpr_scores = np.zeros((train_times, weeks))

    ## for the whole train data set
    pre_t_cv_tpr_scores = np.zeros(train_times)
    pre_t_agg_cv_tpr_scores = np.zeros(train_times)
    t_cv_tpr_scores = np.zeros(train_times)
    t_agg_cv_tpr_scores = np.zeros(train_times)

    ## the last two weeks' data set
    pre_t_eval_cv_tpr_scores = np.zeros(train_times)
    pre_t_eval_agg_cv_tpr_scores = np.zeros(train_times)
    t_eval_cv_tpr_scores = np.zeros(train_times)
    t_eval_agg_cv_tpr_scores = np.zeros(train_times)

    start = time.time()
    skf = StratifiedKFold(n_splits= kfold, random_state= None, shuffle=False)

    for s in range(train_times):
        params['seed'] = s

        pre_cv_train = np.zeros(len(data['train']))
        pre_cv_pred = np.zeros(len(data['test']))
        cv_train = np.zeros(len(data['train']))
        cv_pred = np.zeros(len(data['test']))

        s_start = time.time()

        for w in range(weeks):

            w_start = time.time()
            week_index = data['train'].index[data['train']['wno'] == w]
            week_data = data['train'].iloc[week_index,].reset_index(drop= True)
            week_unlabled_index = week_data.index[week_data['label'] == -1]
            week_data.loc[week_data['label'] == -1, 'label'] = 1

            cv_train_week = np.zeros(len(week_index))
            cv_pred_week = np.zeros(len(data['test']))

            ## pre-train with L and sampling from U with confidence interval, U' is the output of this step
            with utils.timer('Pre-train and PL sampling'):
                # pre-train with CV
                normal_weights = np.ones(len(week_data)).astype('float32')
                normal_weights[week_unlabled_index] = unlabeled_weight
                pre_cv_train_week, pre_cv_pred_week, _, _ = train_with_cv(week_data, data['test'], kfold, skf, params, s, w, normal_weights)

                # pseudo label option1: sample from test data set
                pl_sample_index, remained_index = pseudo_label_sampling(pre_cv_pred_week, pl_sampling_rate, 0.5, 0.5)

                pl_data = data['test'][week_data.columns].iloc[pl_sample_index,].reset_index(drop=True)
                pl_data['label'] = np.array(proba2label(pre_cv_pred_week[pl_sample_index]))

                # ## pseudo label option2: sample from the unlabled data set
                # pl_data = week_data_ul
                # pl_data['label'] = np.array(proba2label(pre_cv_unlabeled_week))

            # ## re-train with L and U'
            # for debug
            pl_iter_cv_train = np.zeros(pl_sampling_times)
            pl_iter_positives_pred = np.zeros(pl_sampling_times)
            ## pseudo label option1
            with utils.timer('Post-train'):
                for it in range(pl_sampling_times):
                    post_week_data = pd.concat([week_data, pl_data], axis=0, ignore_index=True)
                    post_test_data = data['test'].iloc[remained_index,].reset_index(drop=True)

                    # sample weight
                    #pl_weights = np.array([1.0 if(i < len(week_index)) else pl_sample_weight for i in range(len(post_week_data))])
                    pl_weights = np.array(list(normal_weights) + list(np.full(len(pl_data), pl_sample_weight)))

                    post_cv_train_week, post_cv_pred_week, _, _ = train_with_cv(post_week_data, post_test_data, kfold, skf,params, s, w, pl_weights)
                    cv_train_week = post_cv_train_week[:len(week_index)]
                    cv_pred_week[pl_sample_index] = list(post_cv_train_week[len(week_index):])
                    cv_pred_week[remained_index] = list(post_cv_pred_week)

                    assert (len(data['test']) == len(cv_pred_week))

                    pl_iter_cv_train[it] = utils.sum_weighted_tpr(week_data['label'], cv_train_week)
                    pl_iter_positives_pred[it] = np.sum(proba2label(cv_pred_week))

                    # update pl_data
                    if(it < pl_sampling_times - 1):
                        pl_sample_index, remained_index = pseudo_label_sampling(cv_pred_week, pl_sampling_rate, 0.5, 0.5)
                        pl_data = data['test'][week_data.columns].iloc[pl_sample_index,].reset_index(drop= True)
                        pl_data['label'] = np.array(proba2label(pre_cv_pred_week[pl_sample_index]))
            del post_week_data, post_test_data
            gc.collect()

            print('\n==============================')
            print('pseudo labeling with iterative mode, cv on train:')
            print(pl_iter_cv_train.tolist())
            print('pseudo labeling with iterative mode, positives on test:')
            print(pl_iter_positives_pred)
            print('==============================\n')

            # ## pseudo label option2
            # with utils.timer('Post-train'):
            #     post_week_data = pd.concat([week_data, pl_data], axis=0, ignore_index=True)
            #
            #     # sample weight
            #     pl_weights = np.array([1.0 if(i < len(week_index)) else pl_sample_weight for i in range(len(post_week_data))])
            #
            #     post_cv_train_week, post_cv_pred_week, _, _ = train_with_cv(post_week_data, data['test'], kfold, skf,params, s, w, pl_weights)
            #     cv_train_week = post_cv_train_week[:len(week_index)]
            #     cv_pred_week = post_cv_pred_week

            w_end = time.time()
            ## aggregate cv_pred
            pre_cv_pred += list(pre_cv_pred_week)
            cv_pred += list(cv_pred_week)

            pre_cv_train[week_index] = list(pre_cv_train_week)
            cv_train[week_index] = list(cv_train_week)

            week_cv_tpr_scores[s][w] = utils.sum_weighted_tpr(week_data['label'], cv_train_week)
            pre_week_cv_tpr_scores[s][w] = utils.sum_weighted_tpr(week_data['label'], pre_cv_train_week)

            label_positives = np.sum(week_data['label'])
            pre_pred_positives = np.sum(proba2label(pre_cv_train_week))
            pred_positives = np.sum(proba2label(cv_train_week))
            pre_test_positives = np.sum(proba2label(pre_cv_pred_week))
            test_positives = np.sum(proba2label(cv_pred_week))

            print('\n==========================================')
            print('#%s: week %s, original positives %s, pseudo labeled positives %s' % (s, w, np.sum(week_data['label']), np.sum(pl_data['label'])))
            print('#%s: week %s, cv score %.6f/%.6f, positives %s/%s/%s, test positives %s/%s' % (s, w,
                                                                                               pre_week_cv_tpr_scores[s][w], week_cv_tpr_scores[s][w],
                                                                                               label_positives, pre_pred_positives, pred_positives,
                                                                                               pre_test_positives, test_positives))
            print('time elapsed %s' % (int(w_end - w_start)))
            print('==========================================\n')

            del week_data, pl_data, cv_train_week, pre_cv_train_week, cv_pred_week, pre_cv_pred_week
            gc.collect()

            _print_memory_usage()
        s_end = time.time()
        ## average cv_pred by weeks
        pre_cv_pred /= weeks
        cv_pred /= weeks
        pre_final_cv_pred += pre_cv_pred
        final_cv_pred += cv_pred

        pre_final_cv_train += pre_cv_train
        final_cv_train += cv_train

        ## evaluate on whole labeled data
        labeled_index = data['train'].index[data['train']['label'] != -1]
        pre_t_cv_tpr_scores[s] = utils.sum_weighted_tpr(data['train']['label'].iloc[labeled_index,], pre_cv_train[labeled_index])
        pre_t_agg_cv_tpr_scores[s] = utils.sum_weighted_tpr(data['train']['label'].iloc[labeled_index,], pre_final_cv_train[labeled_index]/(s + 1.0))
        t_cv_tpr_scores[s] = utils.sum_weighted_tpr(data['train']['label'].iloc[labeled_index,], cv_train[labeled_index])
        t_agg_cv_tpr_scores[s] = utils.sum_weighted_tpr(data['train']['label'].iloc[labeled_index,], final_cv_train[labeled_index] / (s + 1.0))

        ## for debug
        pre_test_positives = np.sum(proba2label(pre_final_cv_pred/(s + 1.0)))
        test_positives = np.sum(proba2label(final_cv_pred/(s + 1.0)))

        ## evaluate on the last two weeks' labeled data
        eval_index = data['train'].index[(data['train']['wno'] >= weeks - 2) & (data['train']['label'] != -1)]
        pre_t_eval_cv_tpr_scores[s] = utils.sum_weighted_tpr(data['train']['label'].iloc[eval_index,], pre_cv_train[eval_index])
        pre_t_eval_agg_cv_tpr_scores[s] = utils.sum_weighted_tpr(data['train']['label'].iloc[eval_index,], pre_final_cv_train[eval_index]/(s + 1.0))
        t_eval_cv_tpr_scores[s] = utils.sum_weighted_tpr(data['train']['label'].iloc[eval_index,], cv_train[eval_index])
        t_eval_agg_cv_tpr_scores[s] = utils.sum_weighted_tpr(data['train']['label'].iloc[eval_index,], final_cv_train[eval_index]/(s + 1.0))

        print('\n====================================================')
        print('#%s:[@all] current cv score %.6f/%.6f, aggregated cv score %.6f/%.6f, test positives %s/%s' % (s,
                                                                                                     pre_t_cv_tpr_scores[s], t_cv_tpr_scores[s],
                                                                                                     pre_t_agg_cv_tpr_scores[s], t_agg_cv_tpr_scores[s],
                                                                                                     pre_test_positives, test_positives))
        print('#%s:[@partial] current cv score %.6f/%.6f, aggregated cv score %.6f/%.6f' % (s,
                                                                                                        pre_t_eval_cv_tpr_scores[s], t_eval_cv_tpr_scores[s],
                                                                                                        pre_t_eval_agg_cv_tpr_scores[s], t_eval_agg_cv_tpr_scores[s]
                                                                                                        ))
        print('time elapsed %s' % (s_end - s_start))
        print('====================================================\n')

        eval_t_agg_cv_tpr_score_str = ('%.4f' % t_eval_agg_cv_tpr_scores[s]).split('.')[1]
        pre_eval_agg_cv_tpr_score_str = ('%.4f' % pre_t_eval_agg_cv_tpr_scores[s]).split('.')[1]
        t_agg_cv_tpr_score_str = ('%.4f' % t_agg_cv_tpr_scores[s]).split('.')[1]
        pre_t_agg_cv_tpr_score_str = ('%.4f' % pre_t_agg_cv_tpr_scores[s]).split('.')[1]

        pre_score_str = '%s_%s' % (pre_t_agg_cv_tpr_score_str, pre_eval_agg_cv_tpr_score_str)
        score_str = '%s_%s' % (t_agg_cv_tpr_score_str, eval_t_agg_cv_tpr_score_str)

        with utils.timer("model output"):
            OutputDir = '%s/model_b/%s' % (config.DataRootDir, s)
            if (os.path.exists(OutputDir) == False):
                os.makedirs(OutputDir)
            pd.DataFrame({'id': data['train']['id'],
                          'date': data['train']['date'],
                          'score': final_cv_train / (s + 1.0),
                          'label': data['train']['label']}).to_csv('%s/%s_train_%s.csv' % (OutputDir, strategy, score_str), index=False, date_format='%Y%m%d')
            pd.DataFrame({'id': data['test']['id'],
                          'date': data['test']['date'],
                          'score': final_cv_pred / (s + 1.0)}).to_csv('%s/%s_pred_%s.csv' % (OutputDir, strategy, score_str), index=False, date_format='%Y%m%d')
            pd.DataFrame({'id': data['train']['id'],
                          'date': data['train']['date'],
                          'score': pre_final_cv_train / (s + 1.0),
                          'label': data['train']['label']}).to_csv('%s/%s_pre_train_%s.csv' % (OutputDir, strategy, pre_score_str), index=False, date_format='%Y%m%d')
            pd.DataFrame({'id': data['test']['id'],
                          'date': data['test']['date'],
                          'score': pre_final_cv_pred / (s + 1.0)}).to_csv('%s/%s_pre_pred_%s.csv' % (OutputDir, strategy, pre_score_str), index=False, date_format='%Y%m%d')
        _print_memory_usage()

    ## summary
    print('\n================ week cv&eval ===============')
    print('pre-trained week cv tpr scores:')
    print(pre_week_cv_tpr_scores)
    print('week cv tpr scores')
    print(week_cv_tpr_scores)
    print('==============================================\n')

    print('\n========= time current cv&eval ===============')
    print('pre-trained current cv tpr scores:')
    print(pre_t_cv_tpr_scores)
    print('current cv tpr scores')
    print(t_cv_tpr_scores)
    print('==============================================\n')

    print('\n========= time aggregated cv&eval ============')
    print('pre-trained aggregated time cv tpr score:')
    print(pre_t_agg_cv_tpr_scores)
    print('aggregated cv tpr scores')
    print(t_agg_cv_tpr_scores)
    print('==============================================\n')

    end = time.time()
    print('\n------------------------------------')
    print('%s done, time elapsed %ss' % (strategy, int(end - start)))
    print('------------------------------------\n')

## total weeks
weeks = np.max(DataSet['train']['wno']) + 1
print('Weeks %s' % weeks)

_print_memory_usage()

## train and evaluate with local mode
public_train(DataSet, weeks, 4)
