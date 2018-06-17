import pandas as pd
import utils
import os,sys
import config

InputDir = './data/version1/l0/kfold'

valid_dfs = []
for fold in range(5):
    valid = utils.hdf_loader('%s/%s/valid_label.hdf' % (InputDir, fold), 'valid_label').sample(frac= 0.1).reset_index(drop= True)
    valid.to_csv('%s/%s/valid_label_sampled.csv' % (InputDir, fold), index= False)
    test = utils.hdf_loader('%s/%s/test.hdf' % (InputDir, fold), 'test').sample(frac= 0.1).reset_index(drop= True)
    test.to_csv('%s/%s/test_sampled.csv' % (InputDir, fold), index= False)
    print('fold %s done.' % fold)
    valid['fold'] = fold
    valid_dfs.append(valid)
TrainData = pd.concat(valid_dfs, axis= 0, ignore_index= True)
TrainData.to_csv('./data/version1/l0/label_sampled.csv', index= False)
