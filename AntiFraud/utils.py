import pandas as pd
import time, os
from contextlib import contextmanager
from sklearn import metrics
import numpy as np
import config
from matplotlib import pyplot as plt

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
    print(f'\n[{name}] done in {time.time() - t0:.0f} s')

## metric function, sum of weighted TPR
def sum_weighted_tpr(y, scores):
    ''''''
    swt = .0
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label= 1)
    for t in config.tpr_factor.keys():
        swt += config.tpr_factor[t] * tpr[np.where(fpr >= t)[0][0]]
    return swt

## data io
def hdf_saver(data, file, key):
    data.to_hdf(path_or_buf= file, key= key, mode= 'w', complib= 'blosc')

def hdf_loader(file, key):
    return pd.read_hdf(path_or_buf= file, key= key)


def plot_fig(train_results, valid_results, output_dir, model_name):
    if(os.path.exists(output_dir) == False):
        os.makedirs(output_dir)
    colors = ['C%s' % i for i in range(train_results.shape[0])]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Sum of Weighted TPR")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("%s/%s.png" % (output_dir, model_name))
    plt.close()

def TagHoliday(df):
    ''''''
    n = len(df)
    result = ['' for x in range(n)]
    for i in range(n):
        if (i == 0):
            result[i] = 'hid_%s' % 0
        elif ((df[i] - df[i - 1]).days == 1):
            result[i] = result[i - 1]
        else:
            result[i] = 'hid_%s' % (int(result[i - 1].split('_')[1]) + 1)
    return result

def IsTheLast(tags):
    n = len(tags)
    result = []
    for i in range(n - 1):
        if (tags[i] == tags[i + 1]):
            result.append(0)
        else:
            result.append(1)
    result.append(1)
    return result

def IsTheFirst(tags):
    n = len(tags)
    result = []
    for i in range(n):
        if (i == 0):
            result.append(1)
        elif (tags[i] != tags[i - 1]):
            result.append(1)
        else:
            result.append(0)
    return result
