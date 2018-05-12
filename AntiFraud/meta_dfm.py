import config
import utils
import DataReader
import pandas as pd
import numpy as np
from DeepFM import DeepFM
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import sys,os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime

pd.set_option('display.max_rows', None)
# params for DFM
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": config.RANDOM_SEED
}

def _plot_fig(train_results, valid_results, model_name):
    FigOutputDir = '%s/fig' % config.MetaModelOutputDir
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
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
        valid = utils.hdf_loader('%s/valid_label.hdf' % FoldInputDir, 'valid_label')
        valid['fold'] = fold
        valid_dfs.append(valid)
        if(fold == 0):
            TestData = utils.hdf_loader('%s/test.hdf' % FoldInputDir, 'test')
    TrainData = pd.concat(valid_dfs, axis= 0, ignore_index= True)

    return TrainData, TestData

def _run_meta_model_dfm():
    ''''''
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "deepfm"
    elif dfm_params["use_fm"]:
        clf_str = "fm"
    elif dfm_params["use_deep"]:
        clf_str = "dnn"
    strategy = 'meta_%s' % clf_str
    ## loading
    TrainData, TestData = _load_data()
    TestData[strategy] = .0
    ## parser
    num_cols = [c for c in TrainData.columns if(c.startswith('num_'))]
    fd = DataReader.FeatureDictionary(dfTrain=TrainData,
                                      dfTest=TestData,
                                      numeric_cols=num_cols,
                                      ignore_cols=config.IGNORE_COLS)
    parser = DataReader.DataParser(fd)
    ## local CV
    wtpr_results_cv = np.zeros(config.KFOLD, dtype=float)
    wtpr_results_epoch_train = np.zeros((config.KFOLD, dfm_params["epoch"]), dtype=float)
    wtpr_results_epoch_valid = np.zeros((config.KFOLD, dfm_params["epoch"]), dtype=float)
    for fold in range(config.KFOLD):
        FoldData = {
            'train': TrainData[TrainData['fold'] != fold].copy(),
            'valid': TrainData[TrainData['fold'] == fold].copy(),
            'test': TestData.copy()
        }
        ## normalization for numeric features
        with utils.timer('Normalization for the numeric'):
            scaler = MinMaxScaler()
            scaler.fit(np.vstack([np.log1p(FoldData['train'][num_cols].values),
                                  np.log1p(FoldData['valid'][num_cols].values),
                                  np.log1p(FoldData['test'][num_cols].values)]))
            FoldData['train'][num_cols] = scaler.transform(np.log1p(FoldData['train'][num_cols].values))
            FoldData['valid'][num_cols] = scaler.transform(np.log1p(FoldData['valid'][num_cols].values))
            FoldData['test'][num_cols] = scaler.transform(np.log1p(FoldData['test'][num_cols].values))
        ## prepare data for DeepFM
        with utils.timer('Prepare data'):
            Xi_train, Xv_train, y_train = parser.parse(df= FoldData['train'], has_label= True)
            Xi_valid, Xv_valid, y_valid = parser.parse(df= FoldData['valid'], has_label= True)
            Xi_test, Xv_test, ids = parser.parse(df= FoldData['test'], has_label= False)
            dfm_params["feature_size"] = fd.feat_dim
            dfm_params["field_size"] = len(Xi_train[0])
        ## training/inference
        with utils.timer('CV for fold %s' % fold):
            dfm = DeepFM(**dfm_params)
            dfm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)
            print('train done.')
            FoldData['valid'][strategy] = dfm.predict(Xi_valid, Xv_valid)
            wtpr_results_cv[fold] = utils.sum_weighted_tpr(y_valid, FoldData['valid'][strategy])
            FoldData[strategy] = dfm.predict(Xi_test, Xv_test)
            TestData['score'] += FoldData[strategy]
            print('inference done.')
        wtpr_results_epoch_train[fold] = dfm.train_result
        wtpr_results_epoch_valid[fold] = dfm.valid_result
        print('fold %s, weighted tpr %.6f' % (fold, wtpr_results_cv[fold]))
        ## saving
        FoldOutputDir = '%s/kfold/%s' % (config.MetaModelOutputDir, fold)
        if(os.path.exists(FoldOutputDir) == False):
            os.makedirs(FoldOutputDir)
        valid_k = 'valid_label_%s' % strategy
        FoldData['valid'][['label', strategy]].to_csv('%s/%s.hdf' % (FoldOutputDir, valid_k), index= False, float_format= '%.8f')
        test_k = 'test_%s' % strategy
        FoldData['test'][['id', strategy]].to_csv('%s/%s.hdf' % (FoldOutputDir, test_k), index= False, float_format= '%.8f')

    TestData['score'] /= config.KFOLD
    print("%s: %.6f (%.6f)"%(clf_str, wtpr_results_cv.mean(), wtpr_results_cv.std()))

    ## plot
    _plot_fig(wtpr_results_epoch_train, wtpr_results_epoch_valid, strategy)

    ## submit
    SubmitDir = '%s/submit' % config.MetaModelOutputDir
    if(os.path.exists(SubmitDir) == False):
        os.makedirs(SubmitDir)
    filename = "%s_%s_Mean%.5f_Std%.5f.csv"%(strategy, datetime.datetime.now().strftime("%Y-%m-%d"), wtpr_results_cv.mean(), wtpr_results_cv.std())
    TestData[['id','score']].to_csv('%s/%s' % (SubmitDir, filename), index= False, float_format= '%.8f')

if __name__ == '__main__':
    ''''''
    _run_meta_model_dfm()
