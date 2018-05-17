from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import gc,os,sys,datetime,time
import numpy as np
import pandas as pd
import progressbar, psutil
import utils
import config
import DLMCDDataReader
import tensorflow as tf
import dlmcd_utils
from DLMCD import LR, FM, PNN1, PNN2, FNN, CCPM

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if(not p in sys.path):
    sys.path.append(p)
process = psutil.Process(os.getpid())
tf.set_random_seed(config.RANDOM_SEED)
pd.set_option('display.max_rows', None)
strategy = 'dlmcd_%s' % config.dlmcd_params['algo']

def _print_memory_usage():
    ''''''
    print('\n---- Current memory usage %sM ----\n' % int(process.memory_info().rss/(1024*1024)))

def _load_data():
    ''''''
    valid_dfs = []
    for fold in range(config.KFOLD):
        FoldInputDir = '%s/kfold/%s' % (config.MetaModelInputDir, fold)
        if(config.dlmcd_params['debug'] == True):
            sample_frac = 0.05
        else:
            sample_frac = 1.
        valid = utils.hdf_loader('%s/valid_label.hdf' % FoldInputDir, 'valid_label').sample(frac= sample_frac).reset_index(drop= True)
        valid['fold'] = fold
        valid_dfs.append(valid)
        if(fold == 0):
            TestData = utils.hdf_loader('%s/test.hdf' % FoldInputDir, 'test').sample(frac= sample_frac).reset_index(drop= True)
    TrainData = pd.concat(valid_dfs, axis= 0, ignore_index= True)
    return TrainData, TestData

def _get_model(algo):
    if algo == 'lr':
        lr_params = {
            'input_dim': fd.feat_dim,
            'opt_algo': 'gd',
            'learning_rate': 0.1,
            'l2_weight': 0,
            'random_seed': 0
        }
        print(lr_params)
        model = LR(**lr_params)
    elif algo == 'fm':
        fm_params = {
            'input_dim': fd.feat_dim,
            'factor_order': 10,
            'opt_algo': 'gd',
            'learning_rate': 0.1,
            'l2_w': 0,
            'l2_v': 0,
        }
        print(fm_params)
        model = FM(**fm_params)
    elif algo == 'fnn':
        fnn_params = {
            'field_sizes': fd.field_size,
            'embed_size': 10,
            'layer_sizes': [500, 1],
            'layer_acts': ['relu', None],
            'drop_out': [0, 0],
            'opt_algo': 'gd',
            'learning_rate': 0.1,
            'embed_l2': 0,
            'layer_l2': [0, 0],
            'random_seed': 0
        }
        print(fnn_params)
        model = FNN(**fnn_params)
    elif algo == 'ccpm':
        ccpm_params = {
            'field_sizes': fd.field_size,
            'embed_size': 10,
            'filter_sizes': [5, 3],
            'layer_acts': ['relu'],
            'drop_out': [0],
            'opt_algo': 'gd',
            'learning_rate': 0.1,
            'random_seed': 0
        }
        print(ccpm_params)
        model = CCPM(**ccpm_params)
    elif algo == 'pnn1':
        pnn1_params = {
            'field_sizes': fd.field_size,
            'embed_size': 10,
            'layer_sizes': [500, 1],
            'layer_acts': ['relu', None],
            'drop_out': [0, 0],
            'opt_algo': 'gd',
            'learning_rate': 0.1,
            'embed_l2': 0,
            'layer_l2': [0, 0],
            'random_seed': 0
        }
        print(pnn1_params)
        model = PNN1(**pnn1_params)
    elif algo == 'pnn2':
        pnn2_params = {
            'field_sizes': fd.field_size,
            'embed_size': config.dlmcd_params['embedding_size'],
            'layer_sizes': config.dlmcd_params['layer_size'],
            'layer_acts': ['relu', None],
            'drop_out': [0, 0],
            'opt_algo': 'gd',
            'learning_rate': config.dlmcd_params['learning_rate'],
            'embed_l2': 0,
            'layer_l2': [0., 0.],
            'random_seed': config.dlmcd_params['random_seed'],
            'layer_norm': True,
        }
        model = PNN2(**pnn2_params)
    return model

## local CV
start = time.time()
wtpr_results_cv = np.zeros(config.KFOLD, dtype=float)
wtpr_results_epoch_train = np.zeros((config.KFOLD, config.dlmcd_params['num_round']), dtype=float)
wtpr_results_epoch_valid = np.zeros((config.KFOLD, config.dlmcd_params['num_round']), dtype=float)
sub = pd.DataFrame(columns= ['id', 'score'])
for fold in range(config.KFOLD):
    pred_valid = []
    pred_test = []
    label_valid = []
    ## load data
    with utils.timer('Loader'):
        train_data, test_data = _load_data()
    _print_memory_usage()
    ## parser
    with utils.timer('Reader'):
        cate_cols = [c for c in train_data.columns if (c.startswith('cate_'))]
        fd = DLMCDDataReader.FeatureDictionary(dfTrain= train_data[cate_cols], dfTest= test_data[cate_cols],ignore_cols=config.IGNORE_COLS)
        parser = DLMCDDataReader.DataParser(fd)
    _print_memory_usage()
    ##
    #valid_data = train_data[train_data['fold'] == fold]
    #train_data = train_data[train_data['fold'] != fold]
    ## transform
    with utils.timer('Transformer'):
        Xi_train, Xv_train, y_train = parser.parse(df= train_data[train_data['fold'] != fold][['label'] + cate_cols], has_label= True)
        Xi_valid, Xv_valid, y_valid = parser.parse(df= train_data[train_data['fold'] == fold][['label'] + cate_cols], has_label= True)
        label_valid = y_valid.copy()
        #y_train = np.reshape(np.array(y_train), [-1])
        y_train = np.array(y_train).reshape(-1)
        y_valid = np.reshape(np.array(y_valid), [-1])
        del train_data
        gc.collect()
        train_data = dlmcd_utils.libsvm_2_coo(zip(Xi_train, Xv_train), (len(Xi_train), fd.feat_dim)).tocsr(), y_train
        valid_data = dlmcd_utils.libsvm_2_coo(zip(Xi_valid, Xv_valid), (len(Xi_valid), fd.feat_dim)).tocsr(), y_valid
        train_data = dlmcd_utils.shuffle(train_data)
        del Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid
        gc.collect()
    train_size = train_data[0].shape[0]
    valid_size = valid_data[0].shape[0]
    print(train_data[0].shape)
    print(valid_data[0].shape)
    print('train size %s, valid size %s' % (train_size, valid_size))
    ## modelling
    model = _get_model(config.dlmcd_params['algo'])
    for i in range(config.dlmcd_params['num_round']):
        # train
        fetches = [model.optimizer, model.loss]
        with utils.timer('Train'):
            if config.dlmcd_params['batch_size'] > 0:
                ls = []
                bar = progressbar.ProgressBar()
                for j in bar(range(int(train_size / config.dlmcd_params['batch_size'] + 1))):
                    X_i, y_i = dlmcd_utils.slice(train_data, j * config.dlmcd_params['batch_size'], config.dlmcd_params['batch_size'])

                    _, l = model.run(fetches, X_i, y_i)
                    ls.append(l)
            elif config.dlmcd_params['batch_size'] == -1:
                X_i, y_i = dlmcd_utils.slice(train_data)
                _, l = model.run(fetches, X_i, y_i)
                ls = [l]
        _print_memory_usage()
        # evaluate
        with utils.timer('Evaluate'):
            train_preds = []
            valid_preds = []
            bar = progressbar.ProgressBar()
            for j in bar(range(int((train_size / config.dlmcd_params['batch_size']) + 1))):
                X_i, _ = dlmcd_utils.slice(train_data, j * config.dlmcd_params['batch_size'], config.dlmcd_params['batch_size'])
                preds = model.run(model.y_prob, X_i, mode='test')
                train_preds.extend(preds)
            _print_memory_usage()
            bar = progressbar.ProgressBar()
            for j in bar(range(int((valid_size / config.dlmcd_params['batch_size']) + 1))):
                X_i, _ = dlmcd_utils.slice(valid_data, j * config.dlmcd_params['batch_size'], config.dlmcd_params['batch_size'])
                preds = model.run(model.y_prob, X_i, mode='test')
                valid_preds.extend(preds)
            wtpr_results_epoch_train[fold][i] = utils.sum_weighted_tpr(train_data[1], train_preds)
            wtpr_results_epoch_valid[fold][i] = utils.sum_weighted_tpr(valid_data[1], valid_preds)
            if(i == (config.dlmcd_params['num_round'] - 1)):
                pred_valid = valid_preds.copy()
        _print_memory_usage()
        print('[round %d]: loss (with l2 norm) %.6f, train score %.6f, valid score %.6f' % (i, np.mean(ls), wtpr_results_epoch_train[fold][i], wtpr_results_epoch_valid[fold][i]))
    ## predict on test
    with utils.timer('Inference'):
        Xi_test, Xv_test, ids = parser.parse(df= test_data[['id'] + cate_cols], has_label= False)
        test_data = dlmcd_utils.libsvm_2_coo(zip(Xi_test, Xv_test), (len(Xi_test), fd.feat_dim)).tocsr(), ids
        test_size = len(test_data)
        bar = progressbar.ProgressBar()
        for j in bar(range(int((test_size / config.dlmcd_params['batch_size']) + 1))):
            X_i, _ = dlmcd_utils.slice(test_data, j * config.dlmcd_params['batch_size'], config.dlmcd_params['batch_size'])
            pred_test.extends(model.run(model.y_prob, X_i, mode='test'))
    ## saving
    with utils.timer('Saving'):
        FoldOutputDir = '%s/kfold/%s' % (config.MetaModelOutputDir, fold)
        if (os.path.exists(FoldOutputDir) == False):
            os.makedirs(FoldOutputDir)
        valid_k = 'valid_label_%s' % strategy
        out_valid = pd.DataFrame()
        out_valid['label'] = label_valid
        out_valid[strategy] = pred_valid
        out_valid.to_csv('%s/%s.csv' % (FoldOutputDir, valid_k), index=False, float_format='%.8f')
        print('Saving for valid done.')
        test_k = 'test_%s' % strategy
        out_test = pd.DataFrame()
        out_test['id'] = ids
        out_test[strategy] = pred_test
        out_test.to_csv('%s/%s.csv' % (FoldOutputDir, test_k), index=False, float_format='%.8f')
        print('Saving for test done.')

    wtpr_results_cv[fold] = wtpr_results_epoch_valid[fold][-1]
    print('\n=================================')
    print('Fold %s, weighted tpr %.6f' % wtpr_results_cv[fold])
    print('=================================\n')

_print_memory_usage()
sub['score'] /= config.KFOLD
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

## plot
with utils.timer("Plotting"):
    PlotDir = '%s/fig' % config.MetaModelOutputDir
    utils.plot_fig(wtpr_results_epoch_train, wtpr_results_epoch_valid, PlotDir, strategy)

end = time.time()
print('\n------------------------------------')
print('%s done, time elapsed %ss' % (strategy, int(end - start)))
print('------------------------------------\n')