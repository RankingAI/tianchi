import utils
import config
import pandas as pd
import gc,os,sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

strategy = 'meta_lr_dlmcd_fnn'
sub_strategy = 'meta_dlmcd_fnn'
OutputDir = '%s/l1' % config.DataBaseDir

## load numeric features
FeatInputDir = config.MetaModelInputDir
num_dfs = []
num_feats = []
for fold in range(config.KFOLD):
    FoldInputDir = '%s/kfold/%s' % (FeatInputDir, fold)
    valid = utils.hdf_loader('%s/valid_label.hdf' % FoldInputDir, 'valid_label').reset_index(drop= True)
    num_cols = [c for c in valid.columns if(c.startswith('num_'))]
    num_cols = [c for c in num_cols if(c not in config.IGNORE_COLS)]
    valid = valid[num_cols]
    num_dfs.append(valid)
    if(fold == 0):
        num_test = utils.hdf_loader('%s/test.hdf' % FoldInputDir, 'test').reset_index(drop= True)
        num_test = num_test[num_cols]
        num_feats = num_cols.copy()
##
valid_dfs = []
test_dfs = []
InputDir = config.MetaModelOutputDir
for fold in range(config.KFOLD):
    FoldInputDir = '%s/kfold/%s' % (InputDir, fold)
    # for valid
    valid = pd.read_csv('%s/valid_label_%s.csv' % (FoldInputDir, sub_strategy)).reset_index(drop= True)
    for c in num_feats:
        valid[c] = num_dfs[fold][c]
    valid['fold'] = fold
    valid_dfs.append(valid)
    # for test
    test = pd.read_csv('%s/test_%s.csv' % (FoldInputDir, sub_strategy)).reset_index(drop= True)
    for c in num_feats:
        test[c] = num_test[c]
    test_dfs.append(test)
TrainData = pd.concat(valid_dfs, axis= 0, ignore_index= True)
del num_dfs, num_test, valid, test, valid_dfs
gc.collect()

# print(TrainData.head(20))
# print('------------------')
# print(test_dfs[0].head(20))
# print('------------------')
# print(test_dfs[1].head(20))

all_feats = num_feats + [sub_strategy]
## CV
wtpr_results_cv = np.zeros(config.KFOLD, dtype=float)
sub = pd.DataFrame(columns=['id', 'score'])
for fold in range(config.KFOLD):
    FoldData = {
        'train': TrainData[TrainData['fold'] != fold],
        'valid': TrainData[TrainData['fold'] == fold],
    }
    with utils.timer('Normalization for the numeric'):
        scaler = MinMaxScaler()
        scaler.fit(np.vstack([np.log1p(FoldData['train'][num_feats].values),
                              np.log1p(FoldData['valid'][num_feats].values),
                              np.log1p(test_dfs[fold][num_feats].values)]))
        # np.log1p(FoldData['test'][num_feats].values)]))
        FoldData['train'][num_feats] = scaler.transform(np.log1p(FoldData['train'][num_feats].values))
        FoldData['valid'][num_feats] = scaler.transform(np.log1p(FoldData['valid'][num_feats].values))
        test_dfs[fold][num_feats] = scaler.transform(np.log1p(test_dfs[fold][num_feats].values))
    # train
    #weight = np.array(FoldData['train']['label'].values.tolist()) * 9
    model = LogisticRegression(C= 10, max_iter= 100)#, class_weight= {1: 9, 0: 1})
    model.fit(FoldData['train'][all_feats], FoldData['train']['label'])
    # inference
    pred_valid = model.predict_proba(FoldData['valid'][all_feats])[:,1]
    wtpr_results_cv[fold] = utils.sum_weighted_tpr(FoldData['valid']['label'].values, pred_valid)
    pred_test = model.predict_proba(test_dfs[fold][all_feats])[:,1]
    if(fold == 0):
        sub['id'] = test_dfs[fold]['id']
        sub['score'] = .0
    sub['score'] += pred_test
    print('fold %s, weighted tpr %.6f' % (fold, wtpr_results_cv[fold]))
    # saving
    with utils.timer('Saving'):
        FoldOutputDir = '%s/kfold/%s' % (config.MetaModelOutputDir, fold)
        if (os.path.exists(FoldOutputDir) == False):
            os.makedirs(FoldOutputDir)
        valid_k = 'valid_label_%s' % strategy
        out_valid = pd.DataFrame()
        out_valid['label'] = FoldData['valid']['label']
        out_valid[strategy] = pred_valid
        out_valid.to_csv('%s/%s.csv' % (FoldOutputDir, valid_k), index=False, float_format='%.8f')
        print('Saving for valid done.')
        test_k = 'test_%s' % strategy
        out_test = pd.DataFrame()
        out_test['id'] = test_dfs[fold]['id']
        out_test[strategy] = pred_test
        out_test.to_csv('%s/%s.csv' % (FoldOutputDir, test_k), index=False, float_format='%.8f')
        print('Saving for test done.')
print('\n=======================================')
print('CV score %.6f(%.6f)' % (np.mean(wtpr_results_cv), np.std(wtpr_results_cv)))
print('\n=======================================')

sub['score'] /= config.KFOLD
## submit
with utils.timer("Submission"):
    SubmitDir = '%s/submit' % OutputDir
    if (os.path.exists(SubmitDir) == False):
        os.makedirs(SubmitDir)
    filename = "%s_Mean%.5f_Std%.5f.csv" % (strategy, wtpr_results_cv.mean(), wtpr_results_cv.std())
    sub.to_csv('%s/%s' % (SubmitDir, filename), index=False, float_format='%.8f')
