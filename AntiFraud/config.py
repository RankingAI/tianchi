DataBaseDir = './data'
FeatOutputDir = '%s/version1/l0' % DataBaseDir
KFOLD = 5
RANDOM_SEED = 2018

IGNORE_COLS = ['id', 'label', 'date', 'wno', 'fold']
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
