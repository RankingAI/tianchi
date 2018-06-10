import config
import utils
import os,sys,gc,time,datetime,psutil
import numpy as np
import pandas as pd
from sklearn import preprocessing
import lightgbm

process = psutil.Process(os.getpid())
pd.set_option('display.max_rows', None)

# parameters
params = {
    "boosting": "gbdt",
    "objective": "binary",
    "lambda_l2": 2,  # !!!
    'metric': 'auc',

    "num_iterations": config.lgb_params['epoch'],
    "learning_rate": 0.05,  # !!!
    "max_depth": 8,  # !!!
    'scale_pos_weight': 9,
    #'min_data_in_leaf': 50,
    #'num_leaves': 256,
    'cat_smooth': 80,
    'cat_l2': 20,
    #'max_cat_threshold': 64,

    "feature_fraction": 0.9,
    'feature_fraction_seed': 3,
    "bagging_fraction": 0.9,
    'bagging_seed': 3,
    "bagging_freq": 20,
    "min_hessian": 0.001,

    "max_bin": 63,
}

def _print_memory_usage(content):
    ''''''
    print('\n---- Memory usage[%s]: %sM ----\n' % (content, int(process.memory_info().rss/(1024*1024))))

def _load_data():
    ''''''
    valid_dfs = []
    for fold in range(config.KFOLD):
        FoldInputDir = '%s/kfold/%s' % (config.MetaModelInputDir, fold)
        valid = utils.hdf_loader('%s/valid_label.hdf' % FoldInputDir, 'valid_label').reset_index(drop= True)
        if(config.lgb_params['debug'] == True):
            idx_list = [v for v in valid.index.values if((v % 10) == 0)]
            valid = valid.iloc[idx_list, :]
        valid['fold'] = fold
        valid_dfs.append(valid)
        if(fold == 0):
            TestData = utils.hdf_loader('%s/test.hdf' % FoldInputDir, 'test').reset_index(drop= True)
            if(config.lgb_params['debug'] == True):
                idx_list = [v for v in TestData.index.values if((v % 10) == 0)]
                TestData = TestData.iloc[idx_list, :]
    TrainData = pd.concat(valid_dfs, axis= 0, ignore_index= True)

    return TrainData, TestData

def _run_meta_model_lgb():
    start = time.time()
    strategy = 'meta_lgb'
    ## local CV
    wtpr_results_cv = np.zeros(config.KFOLD, dtype=float)
    sub = pd.DataFrame(columns= ['id', 'score'])
    for fold in range(config.KFOLD):
        fold_start = time.time()
        ## load
        with utils.timer('Loader'):
            TrainData, TestData = _load_data()
            _print_memory_usage('after loader')
        ## features
        num_cols = [c for c in TrainData.columns if(c.startswith('num_'))]
        cate_cols = [c for c in TrainData.columns if(c.startswith('cate_'))]
        feats = num_cols.copy()
        feats.extend(cate_cols)
        feats = [c for c in feats if(c not in config.IGNORE_COLS)]
        num_cols = [c for c in feats if(c.startswith('num_'))]
        cate_cols = [c for c in feats if(c.startswith('cate_'))]
        ## label encoding for categorical columns
        with utils.timer('Label encoding'):
            for c in cate_cols:
                lbl = preprocessing.LabelEncoder()
                lbl.fit(np.hstack((TrainData[c], TestData[c])))
                TrainData[c] = lbl.transform(TrainData[c])
                TestData[c] = lbl.transform(TestData[c])
            _print_memory_usage('after label encoding')
        ## normalization for numeric features
        with utils.timer('Normalization for the numeric'):
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(np.vstack([np.log1p(TrainData[num_cols].values),np.log1p(TestData[num_cols].values)]))
            TrainData[num_cols] = scaler.transform(np.log1p(TrainData[num_cols].values))
            TestData[num_cols] = scaler.transform(np.log1p(TestData[num_cols].values))
        _print_memory_usage('after normalization')
        ValidData = TrainData[TrainData['fold'] == fold][feats + ['label']].copy()
        TrainData = TrainData[TrainData['fold'] != fold][feats + ['label']].copy()

        # train
        d_cv = lightgbm.Dataset(TrainData[feats],
                                label= TrainData['label'],
                                silent= True,
                                feature_name= feats,
                                categorical_feature= cate_cols,
                                free_raw_data= True)
        with utils.timer('training'):
            model = lightgbm.train(params, d_cv)
        # inference
        with utils.timer('inference'):
            pred_valid = model.predict(ValidData[feats])
            pred_test = model.predict(TestData[feats])
            if(fold == 0):
                sub['id'] = TestData['id']
                sub['score'] = .0
            sub['score'] += pred_test
            wtpr_results_cv[fold] = utils.sum_weighted_tpr(ValidData['label'], pred_valid)
        ## saving
        with utils.timer('Saving'):
            FoldOutputDir = '%s/kfold/%s' % (config.MetaModelOutputDir, fold)
            if(os.path.exists(FoldOutputDir) == False):
                os.makedirs(FoldOutputDir)
            valid_k = 'valid_label_%s' % strategy
            out_valid = pd.DataFrame()
            out_valid['label'] = ValidData['label']
            out_valid[strategy] = pred_valid
            out_valid.to_csv('%s/%s.csv' % (FoldOutputDir, valid_k), index= False, float_format= '%.8f')
            test_k = 'test_%s' % strategy
            out_test = pd.DataFrame()
            out_test['id'] = TestData['id']
            out_test[strategy] = pred_test
            out_test.to_csv('%s/%s.csv' % (FoldOutputDir, test_k), index= False, float_format= '%.8f')
        print('\n=======================')
        print('Fold %s, weighted trp %.6f' % (fold, wtpr_results_cv[fold]))
        print('=======================\n')
    sub['score'] /= config.KFOLD
    print('cv score list: ')
    print(wtpr_results_cv)
    print('\n===================================')
    print("%s: %.6f (%.6f)"%(strategy, wtpr_results_cv.mean(), wtpr_results_cv.std()))
    print('===================================\n')

    ## submit
    with utils.timer("Submission"):
        SubmitDir = '%s/submit' % config.MetaModelOutputDir
        if(os.path.exists(SubmitDir) == False):
            os.makedirs(SubmitDir)
        filename = "%s_Mean%.5f_Std%.5f.csv"%(strategy, wtpr_results_cv.mean(), wtpr_results_cv.std())
        sub.to_csv('%s/%s' % (SubmitDir, filename), index= False, float_format= '%.8f')

    end = time.time()
    print('\n------------------------------------')
    print('%s done, time elapsed %ss' % (strategy, int(end - start)))
    print('------------------------------------\n')

if __name__ == '__main__':
    ''''''
    _run_meta_model_lgb()