import pandas as pd
import time
from contextlib import contextmanager
from sklearn import metrics
import numpy as np
import config

## timer function
@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

## metric function, sum of weighted TPR
def sum_weighted_tpr(y, scores):
    ''''''
    score = .0
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    for t in config.tpr_factor:
        score += config.tpr_factor[t] * tpr[np.where(fpr >= t)][0][0]
    return score

## data io
def hdf_saver(data, file, key):
    data.to_hdf(path_or_buf= file, key= key, mode= 'w', complib= 'blosc')

def hdf_loader(file, key):
    return pd.read_hdf(path_or_buf= file, key= key)
