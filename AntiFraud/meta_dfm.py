import config
import utils
import DataReader
import pandas as pd
import numpy as np
from scipy.special import erfinv
from DeepFM import DeepFM
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt
import sys,os
import gc
import tensorflow as tf

pd.set_option('display.max_rows', None)
# params for DFM
dfm_params = {
    "use_fm": True,
    "use_deep": False,
    "embedding_size": 8,
    "dropout_fm": [.8, .8],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 10,
    "batch_size": 2048,
    "learning_rate": 0.0008,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.45,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": config.RANDOM_SEED
}

def _plot_fig(train_results, valid_results, model_name):
    FigOutputDir = '%s/fig' % config.MetaModelOutputDir
    if(os.path.exists(FigOutputDir) == False):
        os.makedirs(FigOutputDir)
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

def _run_meta_model_dfm():
    ''''''
    ## local CV
    wtpr_results_cv = np.zeros(config.KFOLD, dtype=float)
    recall_results_cv = np.zeros(config.KFOLD, dtype= float)
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
        ## normalization for numeric features
        with utils.timer('Normalization for numeric features'):
            num_cols = [c for c in DataSet['train_label'].columns if(c.startswith('num_'))]
            # ## RankGauss Normalization
            # DataSet['train_label']['source'] = 'train'
            # DataSet['valid_label']['source'] = 'valid'
            # num_data = pd.concat([DataSet['train_label'][num_cols + ['source']], DataSet['valid_label'][num_cols + ['source']]], ignore_index= True)
            # for c in num_cols:
            #     rank = np.argsort(num_data[c], axis= 0)
            #     uppper = np.max(rank)
            #     lower = np.min(rank)
            #     num_data[c] = (num_data[c] - lower)/(uppper - lower)
            #     num_data[c] = erfinv(num_data[c])
            #     num_data[c] -= np.mean(num_data[c])
            # DataSet['train_label'][num_cols] = num_data[num_data['source'] == 'train'][num_cols].values.copy()
            # DataSet['valid_label'][num_cols] = num_data[num_data['source'] == 'valid'][num_cols].values.copy()
            # del num_data
            # gc.collect()
            # MinMaxScaler/Normalizer
            scaler = MinMaxScaler()
            # scaler = Normalizer()
            scaler.fit(np.vstack([np.log1p(DataSet['train_label'][num_cols].values), np.log1p(DataSet['valid_label'][num_cols].values)]))
            DataSet['train_label'][num_cols] = scaler.transform(np.log1p(DataSet['train_label'][num_cols].values))
            DataSet['valid_label'][num_cols] = scaler.transform(np.log1p(DataSet['valid_label'][num_cols].values))
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
            valid_pred = dfm.predict(Xi_valid, Xv_valid)
            wtpr_results_cv[fold] = utils.weighted_tpr(y_valid, valid_pred)
            recall_results_cv[fold] = utils.recall(y_valid, valid_pred)
            print('fold %s, weighted tpr %.6f, recall %.6f' % (fold, wtpr_results_cv[fold], recall_results_cv[fold]))
        wtpr_results_epoch_train[fold] = dfm.train_result
        wtpr_results_epoch_valid[fold] = dfm.valid_result
        print('-----------------------------------------------------------')

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("Weighted TPR %s: %.6f (%.6f)" % (clf_str, wtpr_results_cv.mean(), wtpr_results_cv.std()))
    print("Recall %s: %.6f (%.6f)" % (clf_str, recall_results_cv.mean(), recall_results_cv.std()))
    filename = "%s_Mean%.6f_Std%.6f.csv"%(clf_str, wtpr_results_cv.mean(), wtpr_results_cv.std())
    #_make_submission(ids_test, y_test_meta, filename)

    _plot_fig(wtpr_results_epoch_train, wtpr_results_epoch_valid, clf_str)

def _make_submission():
    ''''''
    SubmitInputDir = '%s/submit' % config.MetaModelInputDir
    DataSet = {
        'train_label': utils.hdf_loader('%s/train_label.hdf' % SubmitInputDir, 'train_label'),
        'test': utils.hdf_loader('%s/test.hdf' % SubmitInputDir, 'test')
    }
    ## normalization for numeric features
    with utils.timer('Normalization for numeric features'):
        num_cols = [c for c in DataSet['train_label'].columns if (c.startswith('num_'))]
        # MinMaxScaler/Normalizer
        scaler = MinMaxScaler()
        # scaler = Normalizer()
        scaler.fit(np.vstack([np.log1p(DataSet['train_label'][num_cols].values), np.log1p(DataSet['test'][num_cols].values)]))
        DataSet['train_label'][num_cols] = scaler.transform(np.log1p(DataSet['train_label'][num_cols].values))
        DataSet['test'][num_cols] = scaler.transform(np.log1p(DataSet['test'][num_cols].values))
    ## prepare data for DeepFM
    with utils.timer('Prepare data'):
        num_cols = [c for c in DataSet['train_label'].columns if (c.startswith('num_'))]
        fd = DataReader.FeatureDictionary(dfTrain=DataSet['train_label'], dfTest=DataSet['test'], numeric_cols=num_cols, ignore_cols=config.IGNORE_COLS)
        parser = DataReader.DataParser(fd)
        Xi_train, Xv_train, y_train = parser.parse(df=DataSet['train_label'], has_label= True)
        Xi_valid, Xv_valid, ids = parser.parse(df=DataSet['valid_label'], has_label= False)
        dfm_params["feature_size"] = fd.feat_dim
        dfm_params["field_size"] = len(Xi_train[0])
    ## training
    with utils.timer('Training on entire data set'):
        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train, Xv_train, y_train)
    with utils.timer('Inference on test data set'):
        dfm.predict(Xi_valid, Xv_valid)


if __name__ == '__main__':
    ''''''
    _run_meta_model_dfm()
