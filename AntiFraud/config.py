DataBaseDir = './data'
FeatOutputDir = '%s/l0' % DataBaseDir
KFOLD = 3
RANDOM_SEED =  2018

IGNORE_COLS = ['id', 'label', 'date']
NUMERIC_COLS = ['f%s' % i for i in range(82, 87)]
CATEGORICAL_COLS = ['f%s' % i for i in range(1, 298) if(i not in range(82, 87))]

weighted_tpr = {
    0.001: 0.4,
    0.005: 0.3,
    0.01: 0.3
}

# for meta models
MetaModelInputDir = '%s/l0' % DataBaseDir
MetaModelOutputDir = '%s/l1' % DataBaseDir
