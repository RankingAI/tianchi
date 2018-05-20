import pandas as pd
import utils
import os,sys
import config

InputDir = 'data/version1/l1/kfold'

#columns = ['dlmcd_pnn2_%s' % i for i in range(5)]
#merge = pd.DataFrame(columns= [])
sub = pd.DataFrame(columns= ['id', 'score'])
for fold in range(5):
    test = pd.read_csv('%s/%s/test_dlmcd_pnn2.csv' % (InputDir, fold))
    if(fold == 0):
        tmp = pd.read_csv('%s/%s/test_meta_deepfm.csv' % (InputDir, fold))
        sub['id'] = tmp['id']
    sub = sub.merge(test, on= 'id', how= 'left')
    sub.rename(columns= {'dlmcd_pnn2': 'fold_%s' % fold}, inplace= True)

sub['score'] = (sub['fold_0'] + sub['fold_1'] + sub['fold_2'] + sub['fold_3'] + sub['fold_4'])/5
print(sub.isnull().sum(axis= 0))
sub.drop(['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4'], axis= 1, inplace= True)

dfm = pd.read_csv('data/version1/l1/submit/meta_deepfm_20180513_Mean0.43206_Std0.03068.csv')
sub['score'] += dfm['score']
sub['score'] /= 2

## submit
SubmitDir = '%s/submit' % config.MetaModelOutputDir
with utils.timer("Submission"):
    SubmitDir = '%s/submit' % config.MetaModelOutputDir
    if(os.path.exists(SubmitDir) == False):
        os.makedirs(SubmitDir)
    filename = 'avg_dlmcd_dfm.csv'
    sub.to_csv('%s/%s' % (SubmitDir, filename), index= False, float_format= '%.8f')
