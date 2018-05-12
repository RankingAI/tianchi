import sys,os,time,datetime
import numpy as np
import pandas as pd
import config
import utils
from sklearn.model_selection import StratifiedKFold
#import dill as pickle
#import pickle

sys.path.append("..")
pd.set_option('display.max_rows', None)

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

## loading data
cal_dtypes = {
    'dow': 'uint8',
    'is_holiday': 'uint8',
    'is_festival': 'uint8',
    'festival_is_holiday': 'uint8',
    'china_day_of_month': 'uint8'
}
dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
with utils.timer('Load data'):
    DataSet = {
                'train': pd.read_csv('%s/raw/atec_anti_fraud_train.csv' % config.DataBaseDir, parse_dates= ['date'],
                                     date_parser= dateparse),
                'test': pd.read_csv('%s/raw/atec_anti_fraud_test_a.csv' % config.DataBaseDir, parse_dates= ['date'],
                                    date_parser= dateparse),
        'calendar': pd.read_csv('%s/raw/calendar.csv' % config.DataBaseDir, parse_dates=['date'],
                                date_parser=dateparse, dtype=cal_dtypes)
    }
    for mod in ['calendar', 'train', 'test']:
        if (mod == 'calendar'):
            DataSet[mod]['dom'] = DataSet[mod]['date'].dt.day
            DataSet[mod]['dow'] -= 1
            DataSet[mod]['china_day_of_month'] -= 1
        DataSet[mod]['date'] = DataSet[mod]['date'].dt.date
        DataSet[mod].sort_values(by=['date'], inplace=True)

    # ## uniqueness of id
    # assert (len(DataSet['train']['id'].unique()) == len(DataSet['train']))
    # assert (len(DataSet['test']['id'].unique()) == len(DataSet['test']))
## add date features
with utils.timer('Add date features'):
    # add pom/is_weekend
    DataSet['calendar']['pom_lular'] = DataSet['calendar']['china_day_of_month'].apply(
        lambda x: 0 if (x < 10) else(1 if (x < 20) else 2))
    DataSet['calendar']['pom_solar'] = DataSet['calendar']['dom'].apply(
        lambda x: 0 if (x < 10) else(1 if (x < 20) else 2))
    DataSet['calendar'].drop(['china_day_of_month', 'dom'], axis=1, inplace=True)
    DataSet['calendar']['is_weekend'] = DataSet['calendar']['dow'].apply(lambda x: 0 if (x < 5) else 1)
    # add holiday range size
    holidays = DataSet['calendar'][DataSet['calendar']['is_holiday'] == 1][['date']]
    holidays['hol_l0'] = TagHoliday(holidays['date'].values)
    groupped = holidays.groupby(['hol_l0'])
    recs = []
    for g in groupped.groups:
        hol_days = {}
        hol_days['hol_l0'] = g
        hol_days['hol_days'] = len(groupped.get_group(g))
        recs.append(hol_days)
    tmpdf = pd.DataFrame(data=recs, index=range(len(recs)))
    holidays = holidays.merge(tmpdf, how='left', on='hol_l0')
    holidays['last_day_holiday'] = IsTheLast(holidays['hol_l0'].values)
    holidays['first_day_holiday'] = IsTheFirst(holidays['hol_l0'].values)
    DataSet['calendar'] = DataSet['calendar'].merge(holidays, how='left', on='date')
    DataSet['calendar'].drop(['hol_l0'], axis=1, inplace=True)
    DataSet['calendar']['hol_days'].fillna(0, inplace=True)
    DataSet['calendar']['last_day_holiday'].fillna(-1, inplace=True)
    DataSet['calendar']['first_day_holiday'].fillna(-1, inplace=True)
    # add prev/next holiday
    DataSet['calendar']['prevday'] = DataSet['calendar']['date'] - datetime.timedelta(days=1)
    DataSet['calendar']['nextday'] = DataSet['calendar']['date'] + datetime.timedelta(days=1)
    DataSet['calendar'].set_index('date', inplace=True)
    DataSet['calendar']['prev_is_holiday'] = 0
    DataSet['calendar']['prev_is_holiday'] = \
    DataSet['calendar'][DataSet['calendar']['prevday'] >= datetime.datetime(2017, 9, 5).date()]['prevday'].apply(
        lambda x: DataSet['calendar'].loc[x, 'is_holiday'])
    DataSet['calendar']['next_is_holiday'] = 0
    DataSet['calendar']['next_is_holiday'] = \
    DataSet['calendar'][DataSet['calendar']['nextday'] <= datetime.datetime(2018, 2, 5).date()]['nextday'].apply(
        lambda x: DataSet['calendar'].loc[x, 'is_holiday'])
    DataSet['calendar'].reset_index(inplace=True)
    DataSet['calendar'].drop(['prevday', 'nextday'], axis=1, inplace=True)
    DataSet['calendar']['prev_is_holiday'].fillna(0, inplace=True)
    DataSet['calendar']['next_is_holiday'].fillna(0, inplace=True)

## add week no for data split
with utils.timer('Add week no'):
    DataSet['calendar']['wno'] = (DataSet['calendar']['date'].apply(lambda  x: (x - datetime.datetime(2017, 9, 5).date()).days) / 7).astype('int16')

## add missing feature number
with utils.timer('Add missing feat number'):
    for mod in ['train', 'test']:
        DataSet[mod]['num_missing_feat'] = DataSet[mod][config.NUMERIC_COLS + config.CATEGORICAL_COLS].isnull().sum(axis=1)

## renaming
with utils.timer('Rename columns'):
    date_col_map = dict((col, 'cate_%s' % col) for col in DataSet['calendar'].columns if (col not in ['date', 'wno']))
    DataSet['calendar'].rename(index=str, columns= date_col_map, inplace=True)
    cate_col_map = dict((c, 'cate_%s' % c) for c in config.CATEGORICAL_COLS)
    for mod in ['train', 'test']:
        DataSet[mod].rename(index= str, columns= cate_col_map, inplace= True)
    num_col_map = dict((c , 'num_%s' % c) for c in config.NUMERIC_COLS)
    for mod in ['train', 'test']:
        DataSet[mod].rename(index= str, columns= num_col_map, inplace= True)

## merge date data
with utils.timer('Merge date data'):
    for mod in ['train', 'test']:
        DataSet[mod] = DataSet[mod].merge(DataSet['calendar'], on=['date'], how='left')

print('\n ======== Summary ======== ')
## label
uni, count = np.unique(DataSet['train']['label'], return_counts=True)
label_count = dict(zip(uni, count))
print('Empty label rate %.3f' % (label_count[-1] / np.sum(count)))
print('Positive label rate %.3f' % (label_count[1] / np.sum(count)))
print('Negative label rate %.3f' % (label_count[0] / np.sum(count)))

print('------------------------------')
## size/len
for mod in ['train', 'test']:
    data_len = len(DataSet[mod])
    data_size = DataSet[mod].size
    print('%s: data len %s, data size %.2fM' % (mod, data_len, (data_size / 1e6)))

print('------------------------------')
## date
for mod in ['train', 'test']:
    start_date, end_date = DataSet[mod]['date'].min(), DataSet[mod]['date'].max()
    start_wno, end_wno = DataSet[mod]['wno'].min(), DataSet[mod]['wno'].max()
    print('%s: date range %s - %s, wno range %s-%s' % (mod, start_date, end_date, start_wno, end_wno))
print('=======================\n')

num_cols = [c for c in DataSet['train'].columns if(c.startswith('num_'))]
cate_cols = [c for c in DataSet['train'].columns if(c.startswith('cate_'))]
## fill the missing
with utils.timer('Fill the missing'):
    for mod in ['train', 'test']:
        for c in cate_cols:
            DataSet[mod][c].fillna(-1, inplace= True)
            #DataSet[mod][c] = DataSet[mod][c].astype('int32')
        for c in num_cols:
            DataSet[mod][c].fillna(-1, inplace= True)
            #DataSet[mod][c] = DataSet[mod][c].astype('float32')

features = cate_cols.copy()
features.extend(num_cols)
## CV, random split
with utils.timer('CV'):
    skf = StratifiedKFold(n_splits= config.KFOLD, random_state= 2018)
    for fold, (train_index, test_index) in enumerate(skf.split(DataSet['train'][features], DataSet['train']['label'])):
        FoldOutputDir = '%s/kfold/%s' % (config.FeatOutputDir, fold)
        if(os.path.exists(FoldOutputDir) == False):
            os.makedirs(FoldOutputDir)
        FoldData = DataSet['train'].iloc[test_index].copy()
        utils.hdf_saver(
            FoldData[FoldData['label']!= -1][['label'] + features],
            '%s/valid_label.hdf' % FoldOutputDir,
            'valid_label'
        )
        utils.hdf_saver(
            FoldData[FoldData['label'] == -1][['label'] + features],
            '%s/valid_none_label' % FoldOutputDir,
            'valid_none_label'
        )
        utils.hdf_saver(
            DataSet['test'][['id'] + features],
            '%s/test.hdf' % FoldOutputDir,
            'test'
        )
        print('fold %s done.' % fold)

## CV, split residing with time-series
# with utils.timer('CV split saving'):
#     for fold in range(config.KFOLD):
#         FoldOutputDir = '%s/kfold/%s' % (config.FeatOutputDir, fold)
#         if(os.path.exists(FoldOutputDir) == False):
#             os.makedirs(FoldOutputDir)
#         train_start_wno, train_end_wno = DataSet['train']['wno'].min(), DataSet['train']['wno'].max()
#         valid_wno = train_end_wno - fold
#         train_end_wno = valid_wno - 2
#         utils.hdf_saver(
#             DataSet['train'][(DataSet['train']['wno'] == valid_wno) & (DataSet['train']['label'] != -1)][['label'] + features],
#             '%s/valid_label.hdf' % FoldOutputDir,
#             'valid_label'
#         )
#         utils.hdf_saver(
#             DataSet['train'][(DataSet['train']['wno'] == valid_wno) & (DataSet['train']['label'] == -1)][['label'] + features],
#             '%s/valid_none_label.hdf' % FoldOutputDir,
#             'valid_none_label'
#         )
#         utils.hdf_saver(
#             DataSet['train'][(DataSet['train']['wno'] <= train_end_wno) & (DataSet['train']['label'] != -1)][['label'] + features],
#             '%s/train_label.hdf' % FoldOutputDir,
#             'train_label'
#         )
#         utils.hdf_saver(
#             DataSet['train'][(DataSet['train']['wno'] <= train_end_wno) & (DataSet['train']['label'] == -1)][['label'] + features],
#             '%s/train_none_label.hdf' % FoldOutputDir,
#             'train_none_label'
#         )
#         print('fold %s done.' % fold)
# ## submit output
# with utils.timer('Submit saving'):
#     SubmitOutputDir = '%s/submit' % (config.FeatOutputDir)
#     if(os.path.exists(SubmitOutputDir) == False):
#         os.makedirs(SubmitOutputDir)
#     # with open('%s/train_label.pkl' % SubmitOutputDir, 'wb') as f:
#     #     pickle.dump(DataSet['train'][DataSet['train']['label'] != -1][['label'] + features], f)
#     # f.close()
#     # with open('%s/train_none_label.pkl' % SubmitOutputDir, 'wb') as f:
#     #     pickle.dump(DataSet['train'][DataSet['train']['label'] == -1][['label'] + features], f)
#     # f.close()
#     # with open('%s/test.pkl' % SubmitOutputDir, 'wb') as f:
#     #     pickle.dump(DataSet['test'][['id'] + features], f)
#     # f.close()
#     utils.hdf_saver(
#         DataSet['train'][DataSet['train']['label'] != -1][['label'] + features],
#         '%s/train_label.hdf' % SubmitOutputDir,
#         'train_label'
#     )
#     utils.hdf_saver(
#         DataSet['train'][DataSet['train']['label'] == -1][['label'] + features],
#         '%s/train_none_label.hdf' % SubmitOutputDir,
#         'train_none_label'
#     )
#     utils.hdf_saver(
#         DataSet['test'][['id'] + features],
#         '%s/test.hdf' % SubmitOutputDir,
#         'test'
#     )
