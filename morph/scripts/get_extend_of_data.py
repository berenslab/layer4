import pandas as pd
import numpy as np

DATASET_DESCRIPTION = "./data/dataset_description.csv"

dataset = pd.read_csv(DATASET_DESCRIPTION).rename(columns={'Unnamed: 0': 'cell_no'})

full_max_extend = np.zeros((3,))
full_min_extend = np.zeros((3,))

axon_max_extend = np.zeros((3,))
axon_min_extend = np.zeros((3,))

dendrite_max_extend = np.zeros((3,))
dendrite_min_extend = np.zeros((3,))

for n, item in dataset.iterrows():
    # adjust file path here	
    nt = pd.read_csv("./data/nt/%s_%s.swc" % (item['type'], item['swc'][:-4]),
                     sep=" ", names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'])
    max_ = np.max(nt[['x', 'y', 'z']])
    min_ = np.min(nt[['x', 'y', 'z']])

    full_max_extend = np.max((full_max_extend, max_), axis=0)
    full_min_extend = np.min((full_min_extend, min_), axis=0)

    if not nt[nt['type'] == 2].empty:
        max_ = np.max(nt[nt['type'] == 2][['x', 'y', 'z']])
        min_ = np.min(nt[nt['type'] == 2][['x', 'y', 'z']])

        axon_max_extend = np.max((axon_max_extend, max_), axis=0)
        axon_min_extend = np.min((axon_min_extend, min_), axis=0)
    else:
        print("cell %i has no axon" % n)

    if not nt[nt['type'] == 3].empty:
        max_ = np.max(nt[nt['type'] == 3][['x', 'y', 'z']])
        min_ = np.min(nt[nt['type'] == 3][['x', 'y', 'z']])

        dendrite_max_extend = np.max((dendrite_max_extend, max_), axis=0)
        dendrite_min_extend = np.min((dendrite_min_extend, min_), axis=0)
    else:
        print("cell %i has no dendrites"% n)

f = np.vstack((full_min_extend, full_max_extend))
a = np.vstack((axon_min_extend, axon_max_extend))
d = np.vstack((dendrite_min_extend, dendrite_max_extend))
data = np.hstack((f, a, d))

extend = pd.DataFrame(data.T, index=['full_width', 'full_height', 'full_depth', 'axon_width', 'axon_height',
                                     'axon_depth', 'dendrite_width', 'dendrite_height', 'dendrite_depth'],
                      columns=['min', 'max'])
# adjust file path here
extend.to_csv("./data/cell_extend.csv")
