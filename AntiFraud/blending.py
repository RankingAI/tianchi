import pandas as pd

DataBaseDir = 'data/submit_b'

blending_weights = {
    'fanrongmeng': 0.1,
    'hengwang': 0.25,
    'jianyuzhou': 0.1,
    'pingzhouyuan1': 0.3, # lgb with PL
    'pingzhouyuan2': 0.15, #
    'yanglu': 0.1,
}

##
DataSet = {
    'fanrongmeng': pd.read_csv('%s/submit_rawFea_lgb_039_20180703_080619.csv' % DataBaseDir).sort_values(by= ['date'])[['id', 'score']].reset_index(drop= True),
    'hengwang': pd.read_csv('%s/7-8result.csv' % DataBaseDir).sort_values(by= ['date'])[['id', 'score']].reset_index(drop= True),
    'jianyuzhou': pd.read_csv('%s/lgb18ensemble date-v9.csv' % DataBaseDir).sort_values(by= ['date'])[['id', 'score']].reset_index(drop= True),
    'pingzhouyuan1': pd.read_csv('%s/sub_lgb_sss_pred_3494_3214.csv' % DataBaseDir)[['id', 'score']],
    'pingzhouyuan2': pd.read_csv('%s/sub_lgb_sss_pre_pred_4009_3824.csv' % DataBaseDir)[['id', 'score']],
    'yanglu': pd.read_csv('%s/test_01_0628_63models_nofillna_4015_b.csv' % DataBaseDir).sort_values(by= ['date'])[['id', 'score']].reset_index(drop= True),
}

print(DataSet['fanrongmeng'].head(10))
print('------------------------')
print(DataSet['hengwang'].head(10))
print('------------------------')
print(DataSet['jianyuzhou'].head(10))
print('------------------------')
print(DataSet['pingzhouyuan1'].head(10))
print('------------------------')
print(DataSet['pingzhouyuan2'].head(10))
print('------------------------')
print(DataSet['yanglu'].head(10))

blending = pd.DataFrame()
blending['id'] = DataSet['pingzhouyuan1']['id']
blending['score'] = .0
for s in blending_weights.keys():
    blending['score'] += blending_weights[s] * DataSet[s]['score']
    print(s)
print('-----------------------')
print(blending.head(10))

weight_numbers = []
for s in blending_weights.keys():
    weight_numbers.append(int(blending_weights[s] * 100))
version_number = '_'.join(['%s' % wn for wn in weight_numbers])
blending.to_csv('%s/blending_%s.csv' % (DataBaseDir, version_number), index= False)
