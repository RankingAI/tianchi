import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

data = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]])
y = np.array([1, 0, 1, 1, 0])
df = pd.DataFrame(data, columns = ['col1', 'col2'])
skf = StratifiedKFold(n_splits=3, random_state= None, shuffle= True)

print(data.shape)
print(y.shape)

d_mat = np.hstack([data, y.reshape(-1, 1)])
for i in range(5):
    #np.random.shuffle(d_mat)
    for fold, (tra_index, val_index) in enumerate(skf.split(d_mat[:,:-1], d_mat[:,-1])):
        print('--------- %s fold %s' % (i, fold))
        print(val_index)
        print(df.iloc[val_index, ])
