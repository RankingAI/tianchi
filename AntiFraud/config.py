DataBaseDir = './data'
FeatOutputDir = '%s/version1/l0' % DataBaseDir
KFOLD = 5
RANDOM_SEED = 2018

IGNORE_COLS = ['id', 'label', 'date', 'wno', 'fold', 'num_is_festival', 'num_festival_is_holiday',
               'num_china_day_of_month']
NUMERIC_COLS = ['f%s' % i for i in range(82, 87)]
CATEGORICAL_COLS = ['f%s' % i for i in range(1, 298) if(i not in range(82, 87))]

tpr_factor = {
    0.001: 0.4,
    0.005: 0.3,
    0.01: 0.3
}

# for meta models
MetaModelInputDir = '%s/version1/l0' % DataBaseDir
MetaModelOutputDir = '%s/version1/l1' % DataBaseDir

# for dfm
dfm_params = {
    'batch_norm_decay': 0.995,
}

# for auto encoder
ae_params = {
    'encoder_layers': [100],
    'epochs': 10,
    'batch_size': 256,
    'display_step': 1,
    'learning_rate': 0.0001,
    'random_seed': 2018,
    'verbose': True,
    'phase': 'train',
    'debug': True,
}

# for dlmcd
dlmcd_params = {
    'algo': 'pnn2',
    'min_round': 1,
    'num_round': 5,
    'early_stop_round': 5,
    'batch_size': 256,
    'debug': True,
    'embedding_size': 8,
    'layer_size': [200, 1],
    'learning_rate': 0.1,
    'random_seed': 2018
}
