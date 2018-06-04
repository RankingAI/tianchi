import utils
import config
import pandas as pd
import gc,os
import numpy as np
from sklearn.linear_model import LogisticRegression

strategy = 'l1_lr'
meta_strategies = ['meta_deepfm', 'meta_dlmcd_fnn']#, 'meta_dlmcd_pnn2']
InputDir = config.MetaModelOutputDir
OutputDir = '%s/l2' % config.DataBaseDir
valid_dfs = []
test_dfs = []
for fold in range(config.KFOLD):
    FoldInputDir = '%s/kfold/%s' % (InputDir, fold)
    fold_dfs = []
    fold_label = []
    with utils.timer('fold %s' % fold):
        # for valid
        for i in range(len(meta_strategies)):
            valid = pd.read_csv('%s/valid_label_%s.csv' % (FoldInputDir, meta_strategies[i]))
            fold_dfs.append(valid[[meta_strategies[i]]])
            if(i == 0):
                fold_label = valid['label'].values.tolist()
        FoldValid = pd.concat(fold_dfs, axis= 1)
        FoldValid['label'] = fold_label
        FoldValid['fold'] = fold
        valid_dfs.append(FoldValid)
        # for test
        fold_dfs = []
        fold_ids = []
        for i in range(len(meta_strategies)):
            test = pd.read_csv('%s/test_%s.csv' % (FoldInputDir, meta_strategies[i]))
            fold_dfs.append(test[[meta_strategies[i]]])
            if(i == 0):
                fold_ids = test['id'].values.tolist()
        FoldTest = pd.concat(fold_dfs, axis= 1)
        FoldTest['id'] = fold_ids
        test_dfs.append(FoldTest)
TrainData = pd.concat(valid_dfs, axis= 0, ignore_index= True)
del valid_dfs, FoldValid, FoldTest
gc.collect()

#print(TrainData.head(20))
#print('----------------------------')
#print(test_dfs[0].head(20))
#print('----------------------------')
#print(test_dfs[1].head(20))

## CV
wtpr_results_cv = np.zeros(config.KFOLD, dtype=float)
sub = pd.DataFrame(columns=['id', 'score'])
for fold in range(config.KFOLD):
    FoldData = {
        'train': TrainData[TrainData['fold'] != fold],
        'valid': TrainData[TrainData['fold'] == fold],
    }
    # train
    #weight = np.array(FoldData['train']['label'].values.tolist()) * 9
    model = LogisticRegression(C= 0.01, max_iter= 20)#, class_weight= {1: 9, 0: 1})
    model.fit(FoldData['train'][meta_strategies], FoldData['train']['label'])
    # inference
    pred_valid = model.predict_proba(FoldData['valid'][meta_strategies])[:,1]
    wtpr_results_cv[fold] = utils.sum_weighted_tpr(FoldData['valid']['label'].values, pred_valid)
    pred_test = model.predict_proba(test_dfs[fold][meta_strategies])[:,1]
    if(fold == 0):
        sub['id'] = test_dfs[fold]['id']
        sub['score'] = .0
    sub['score'] += pred_test
    print('fold %s, weighted tpr %.6f' % (fold, wtpr_results_cv[fold]))
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
