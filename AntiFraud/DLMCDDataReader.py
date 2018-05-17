"""
A data parser for ATEC Anti Fraud Detection competition's dataset.
URL: https://dc.cloud.alipay.com/index#/topic/intro?id=4
"""
import pandas as pd
import numpy as np

## for Multi-field Categorical Data
class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,dfTrain=None, dfTest=None, ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        self.fields = [c for c in df.columns if(c not in self.ignore_cols)]
        self.field_size = [0] * len(self.fields)
        for i in range(len(self.fields)):
            us = df[self.fields[i]].unique()
            self.feat_dict[self.fields[i]] = dict(zip(us, range(np.sum(self.field_size[:i]).astype('int32'),
                                                                (np.sum(self.field_size[:i])).astype('int32') + len(us))))
            self.field_size[i] = len(us)
        self.field_offset = [np.sum(self.field_size[:i]).astype('int32') for i in range(len(self.field_size))]
        self.feat_dim = np.sum(self.field_size)


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            y = dfi['label'].values.tolist()
            dfi.drop(['label'], axis=1, inplace=True)
        else:
            ids = dfi['id'].values.tolist()
            dfi.drop(['id'], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
            dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids

