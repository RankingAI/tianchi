import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime, sys, time
import config
from contextlib import contextmanager
pd.set_option('display.max_rows', None)

debug = True
DataBase = './data'

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

def plotting_line(data, count_dict, start, end, size):
    ''''''
    select_list = [(pair[0], pair[1]) for pair in count_dict.items() if((pair[1] > start) & (pair[1] <= end))]
    sorted_list = sorted(select_list, key= lambda x: x[1], reverse= True)
    feat_name = [p[0] for p in sorted_list]
    fig, axes = plt.subplots(len(feat_name), 1, figsize=(16, size), dpi=100)
    for i in range(len(feat_name)):
        fn = feat_name[i]
        #f_df = data.groupby(fn)['label'].agg([('sum', np.sum), ('count', len)]).reset_index()
        fn_values = sorted(list(np.unique(data[fn])))
        fn_ratios = []
        for v in fn_values:
            tmp_df = data[data[fn] >= v]
            fn_ratios.append(np.sum(tmp_df['label'])/len(tmp_df))
        x = fn_values
        y = fn_ratios

        line1, = axes[i].plot(x, y, lw=2, marker='*', color='r')
        axes[i].legend([line1], ['%s: %s' % (fn, count_dict[fn])], loc=1)
        axes[i].grid()
        axes[i].set_ylabel('%s ratio' % fn)
        print('column %s done.' % fn)
    plt.savefig('%s_%s.png' % (start, end))

if __name__ == '__main__':
    ''''''
    with timer('load data'):
        ## loading data
        cal_dtypes = {
            'dow': 'uint8',
            'is_holiday': 'uint8'
        }
        dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
        DataSet = {
            'train': pd.read_csv('%s/raw/atec_anti_fraud_train.csv' % config.DataRootDir, parse_dates=['date'],
                                 date_parser=dateparse),
            'test': pd.read_csv('%s/raw/atec_anti_fraud_test_a.csv' % config.DataRootDir, parse_dates=['date'],
                                 date_parser=dateparse),
        }
        if(debug == True):
            tra_idx_list = [v for v in DataSet['train'].index.values if ((v % 10) == 0)]
            DataSet['train'] = DataSet['train'].iloc[tra_idx_list, :].reset_index(drop=True)
            tes_idx_list = [v for v in DataSet['test'].index.values if ((v % 10) == 0)]
            DataSet['test'] = DataSet['test'].iloc[tes_idx_list, :].reset_index(drop=True)

    features = [c for c in DataSet['train'].columns if(c.startswith('f'))]
    entire_data = pd.concat([DataSet['train'][features], DataSet['test'][features]], axis= 0).reset_index(drop= True)
    total_size = len(entire_data)
    null_dict = {}
    for feat in features:
        ratio = entire_data[feat].isnull().sum()/total_size
        null_dict[feat] = ratio
    group_null_dict = {}
    for k in null_dict.keys():
        if(null_dict[k] not in group_null_dict):
            group_null_dict[null_dict[k]] = []
        group_null_dict[null_dict[k]].append(k)
    sorted_list = sorted(group_null_dict.items(), key= lambda x: x[0], reverse= True)
    for sl in sorted_list:
        print(sl[0], sl[1])
    # with timer('value count'):
    #     feats = [c for c in data.columns if (c.startswith('f'))]
    #     #cate_feats = [c for c in feats if(c.startswith('cate_'))]
    #     count_dict = {}
    #     for c in feats:
    #         vc = data[c].value_counts()
    #         count_dict[c] = len(vc)
    #
    # with timer('plotting for 0-10'):
    #     plotting_line(data, count_dict, 0, 10, 100)
    # with timer('plotting for 10-100'):
    #     plotting_line(data, count_dict, 10, 100, 200)
    # with timer('plotting for 100-500'):
    #     plotting_line(data, count_dict, 100, 500, 200)
    # with timer('plotting for 500-inf'):
    #     plotting_line(data, count_dict, 500, np.inf, 50)

