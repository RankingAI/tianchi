import config
import utils
import DataReader
import pandas as pd
import numpy as np
from DeepFM import DeepFM
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import sys,os,psutil
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime, time
import gc

process = psutil.Process(os.getpid())
tf.set_random_seed(config.RANDOM_SEED)
pd.set_option('display.max_rows', None)
# params for DFM
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [0.8, 0.8],
    "deep_layers": [16, 16],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 7,
    "batch_size": 1024,
    "learning_rate": 0.0008,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": config.dfm_params['batch_norm_decay'],
    "l2_reg": 0.45,
    "verbose": True,
    "eval_metric": utils.sum_weighted_tpr,
    "random_seed": config.RANDOM_SEED
}

def _print_memory_usage():
    ''''''
    print('\n---- Current memory usage %sM ----\n' % int(process.memory_info().rss/(1024*1024)))

def _plot_fig(train_results, valid_results, model_name):
    FigOutputDir = '%s/fig' % config.MetaModelOutputDir
    if(os.path.exists(FigOutputDir) == False):
        os.makedirs(FigOutputDir)
    #colors = ["red", "blue", "green"]
    colors = ['C%s' % i for i in range(train_results.shape[0])]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Sum of Weighted TPR")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("%s/%s.png" % (FigOutputDir, model_name))
    plt.close()

def _run_meta_model_dfm_ts():
    ''''''
    ## local CV
    wtpr_results_cv = np.zeros(config.KFOLD, dtype=float)
    wtpr_results_epoch_train = np.zeros((config.KFOLD, dfm_params["epoch"]), dtype=float)
    wtpr_results_epoch_valid = np.zeros((config.KFOLD, dfm_params["epoch"]), dtype=float)
    for fold in range(config.KFOLD):
        FoldInputDir = '%s/kfold/%s' % (config.MetaModelInputDir, fold)
        ## load data
        with utils.timer('Load data'):
            DataSet = {
                'train_label': utils.hdf_loader('%s/train_label.hdf' % FoldInputDir, 'train_label'),
                #'train_none_label': utils.hdf_loader('%s/train_none_label.hdf' % FoldInputDir, 'train_none_label'),
                'valid_label': utils.hdf_loader('%s/valid_label.hdf' % FoldInputDir, 'valid_label'),
                #'valid_none_label': utils.hdf_loader('%s/valid_none_label.hdf' % FoldInputDir, 'valid_none_label')
            }
        ## prepare data for DeepFM
        with utils.timer('Prepare data'):
            num_cols = [c for c in DataSet['train_label'].columns if(c.startswith('num_'))]
            fd = DataReader.FeatureDictionary(dfTrain= DataSet['train_label'],
                                              dfTest= DataSet['valid_label'],
                                              numeric_cols= num_cols,
                                              ignore_cols= config.IGNORE_COLS)
            parser = DataReader.DataParser(fd)
            Xi_train, Xv_train, y_train = parser.parse(df= DataSet['train_label'], has_label= True)
            Xi_valid, Xv_valid, y_valid = parser.parse(df= DataSet['valid_label'], has_label= True)
            dfm_params["feature_size"] = fd.feat_dim
            dfm_params["field_size"] = len(Xi_train[0])
        ## training
        with utils.timer('Training on fold %s' % fold):
            dfm = DeepFM(**dfm_params)
            dfm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)
        ## inference
        with utils.timer('Inference on fold %s' % fold):
            wtpr_results_cv[fold] = utils.sum_weighted_tpr(y_valid, dfm.predict(Xi_valid, Xv_valid))
        wtpr_results_epoch_train[fold] = dfm.train_result
        wtpr_results_epoch_valid[fold] = dfm.valid_result
        print('fold %s, weighted tpr %.6f' % (fold, wtpr_results_cv[fold]))

        break

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, wtpr_results_cv.mean(), wtpr_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, wtpr_results_cv.mean(), wtpr_results_cv.std())
    #_make_submission(ids_test, y_test_meta, filename)

    _plot_fig(wtpr_results_epoch_train, wtpr_results_epoch_valid, clf_str)
    #return y_train_meta, y_test_meta

def _load_data():
    ''''''
    valid_dfs = []
    for fold in range(config.KFOLD):
        FoldInputDir = '%s/kfold/%s' % (config.MetaModelInputDir, fold)
        valid = utils.hdf_loader('%s/valid_label.hdf' % FoldInputDir, 'valid_label').reset_index(drop= True)
        #valid = pd.read_csv('%s/valid_label.csv' % FoldInputDir)
        valid['fold'] = fold
        valid_dfs.append(valid)
        if(fold == 0):
            #TestData = pd.read_csv('%s/test.csv' % FoldInputDir)
            TestData = utils.hdf_loader('%s/test.hdf' % FoldInputDir, 'test').reset_index(drop= True)
    TrainData = pd.concat(valid_dfs, axis= 0, ignore_index= True)

    return TrainData, TestData

def _run_meta_model_dfm():
    ''''''
    start = time.time()
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "deepfm"
    elif dfm_params["use_fm"]:
        clf_str = "fm"
    elif dfm_params["use_deep"]:
        clf_str = "dnn"
    strategy = 'meta_%s' % clf_str
    ## local CV
    wtpr_results_cv = np.zeros(config.KFOLD, dtype=float)
    wtpr_results_epoch_train = np.zeros((config.KFOLD, dfm_params["epoch"]), dtype=float)
    wtpr_results_epoch_valid = np.zeros((config.KFOLD, dfm_params["epoch"]), dtype=float)
    sub = pd.DataFrame(columns= ['id', 'score'])
    for fold in range(config.KFOLD):
        fold_start = time.time()
        ## load
        with utils.timer('Loader'):
            _print_memory_usage()
            TrainData, TestData = _load_data()
            _print_memory_usage()
        ## features
        num_cols = [c for c in TrainData.columns if(c.startswith('num_'))]
        cate_cols = [c for c in TrainData.columns if(c.startswith('cate_'))]
        feats = num_cols.copy()
        feats.extend(cate_cols)
        ## parser
        with utils.timer('Reader'):
            fd = DataReader.FeatureDictionary(dfTrain= TrainData[feats], dfTest= TestData[feats], numeric_cols=num_cols, ignore_cols=config.IGNORE_COLS)
            parser = DataReader.DataParser(fd)
            FoldData = {
                'train': TrainData[TrainData['fold'] != fold].copy(),
                'valid': TrainData[TrainData['fold'] == fold].copy(),
                #'test': TestData.copy()
            }
            _print_memory_usage()
            del TrainData#, TestData
            gc.collect()
            print('fraud ratio %.6f/%.6f' % ((FoldData['train']['label'].sum()/len(FoldData['train'])), (FoldData['valid']['label'].sum()/len(FoldData['valid']))))
            print('train length %s, valid length %s' % (len(FoldData['train']), len(FoldData['valid'])))
            _print_memory_usage()
        ## normalization for numeric features
        with utils.timer('Normalization for the numeric'):
            scaler = MinMaxScaler()
            scaler.fit(np.vstack([np.log1p(FoldData['train'][num_cols].values),
                                  np.log1p(FoldData['valid'][num_cols].values),
                                  np.log1p(TestData[num_cols].values)]))
                                  #np.log1p(FoldData['test'][num_cols].values)]))
            FoldData['train'][num_cols] = scaler.transform(np.log1p(FoldData['train'][num_cols].values))
            FoldData['valid'][num_cols] = scaler.transform(np.log1p(FoldData['valid'][num_cols].values))
            TestData[num_cols] = scaler.transform(np.log1p(TestData[num_cols].values))
            #FoldData['test'][num_cols] = scaler.transform(np.log1p(FoldData['test'][num_cols].values))
            _print_memory_usage()
        ## parser
        with utils.timer('Parser'):
            Xi_train, Xv_train, y_train = parser.parse(df= FoldData['train'][['label'] + feats], has_label= True)
            Xi_valid, Xv_valid, y_valid = parser.parse(df= FoldData['valid'][['label'] + feats], has_label= True)
            dfm_params["feature_size"] = fd.feat_dim
            dfm_params["field_size"] = len(Xi_train[0])
            _print_memory_usage()
            #del FoldData['train'], FoldData['valid']
            del FoldData
            gc.collect()
            print('feature size %s, field size for train %s' % (dfm_params['feature_size'], dfm_params['field_size']))
            _print_memory_usage()
        ## training/inference
        with utils.timer('CV for fold %s' % fold):
            dfm = DeepFM(**dfm_params)
            dfm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)
            _print_memory_usage()
            del Xi_train, Xv_train
            gc.collect()
            _print_memory_usage()
            print('train done.')
            pred_valid = dfm.predict(Xi_valid, Xv_valid)
            _print_memory_usage()
            del Xi_valid, Xv_valid
            gc.collect()
            _print_memory_usage()
            wtpr_results_cv[fold] = utils.sum_weighted_tpr(y_valid, pred_valid)
            Xi_test, Xv_test, ids = parser.parse(df= TestData[['id'] + feats])
            #Xi_test, Xv_test, ids = parser.parse(df= FoldData['test'][['id'] + feats])
            _print_memory_usage()
            del TestData
            #del FoldData['test']
            gc.collect()
            _print_memory_usage()
            pred_test = dfm.predict(Xi_test, Xv_test)
            _print_memory_usage()
            del Xi_test, Xv_test
            gc.collect()
            _print_memory_usage()
            if(fold == 0):
                sub['id'] = ids
                sub['score'] = .0
            sub['score'] += pred_test
            print('inference done.')
            wtpr_results_epoch_train[fold] = dfm.train_result
            wtpr_results_epoch_valid[fold] = dfm.valid_result
            print('\n===================================')
            print('Fold %s, weighted tpr %.6f' % (fold, wtpr_results_cv[fold]))
            print('===================================\n')
        ## saving
        with utils.timer('Saving'):
            del parser, dfm
            gc.collect()
            _print_memory_usage()
            FoldOutputDir = '%s/kfold/%s' % (config.MetaModelOutputDir, fold)
            if(os.path.exists(FoldOutputDir) == False):
                os.makedirs(FoldOutputDir)
            valid_k = 'valid_label_%s' % strategy
            out_valid = pd.DataFrame()
            out_valid['label'] = y_valid
            out_valid[strategy] = pred_valid
            out_valid.to_csv('%s/%s.csv' % (FoldOutputDir, valid_k), index= False, float_format= '%.8f')
            print('Saving for valid done.')
            test_k = 'test_%s' % strategy
            out_test = pd.DataFrame()
            out_test['id'] = ids
            out_test[strategy] = pred_test
            out_test.to_csv('%s/%s.csv' % (FoldOutputDir, test_k), index= False, float_format= '%.8f')
            print('Saving for test done.')
            _print_memory_usage()
            del out_test, out_valid
            gc.collect()
            _print_memory_usage()
        fold_end = time.time()
        print('Fold %s done, time elapsed %ss' % (fold, int(fold_end - fold_start)))

    _print_memory_usage()
    sub['score'] /= config.KFOLD
    print('\n===================================')
    print("%s: %.6f (%.6f)"%(clf_str, wtpr_results_cv.mean(), wtpr_results_cv.std()))
    print('===================================\n')

    ## submit
    with utils.timer("Submission"):
        SubmitDir = '%s/submit' % config.MetaModelOutputDir
        if(os.path.exists(SubmitDir) == False):
            os.makedirs(SubmitDir)
        filename = "%s_%s_Mean%.5f_Std%.5f.csv"%(strategy, datetime.datetime.now().strftime("%Y-%m-%d"), wtpr_results_cv.mean(), wtpr_results_cv.std())
        sub.to_csv('%s/%s' % (SubmitDir, filename), index= False, float_format= '%.8f')

    ## plot
    with utils.timer("Plotting"):
        _plot_fig(wtpr_results_epoch_train, wtpr_results_epoch_valid, strategy)

    end = time.time()
    print('\n------------------------------------')
    print('%s done, time elapsed %ss' % (strategy, int(end - start)))
    print('------------------------------------\n')


if __name__ == '__main__':
    ''''''
    _run_meta_model_dfm()
