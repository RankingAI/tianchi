import pandas as pd
import time
from contextlib import contextmanager
from sklearn import metrics
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

## metric function
def weighted_tpr(y, scores):
    ''''''
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label= 1)
    # print(fpr[:100])
    # print(tpr[:100])
    # print('--------------------------------')
    # print(fpr[-100:])
    # print(tpr[-100:])
    metric_map = dict(zip(fpr, tpr))
    wtpr = .0
    for t in config.weighted_tpr:
        if(t in metric_map):
            wtpr += (config.weighted_tpr[t] * metric_map[t])
            print(t, metric_map[t])
    return wtpr

##
def recall(y, scores):
    pred = [1 if(s > 0.5) else 0 for s in scores]
    return metrics.recall_score(y, pred)

## data io
def hdf_saver(data, file, key):
    data.to_hdf(path_or_buf= file, key= key, mode= 'w', complib= 'blosc')

def hdf_loader(file, key):
    return pd.read_hdf(path_or_buf= file, key= key)
