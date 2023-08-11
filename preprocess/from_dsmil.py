import glob
import pickle
import os
import random
import csv
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle

# 총 1046
PATH_DATA = '/nfs/phastos/shared/tcga-dataset/tcga_lung_data_feats/'
PATH_LABEL = '/nfs/phastos/shared/tcga-dataset/TCGA.csv'
PATH_PKL = '/mnt/aitrics_ext/ext01/shared/'
FOLDNUM = 5
NUM_CLASS = 2
SEED = 1
MAP = {0: 'AD', 1: 'SC'}

NUM_TR = 835
NUM_FOLD = 167

pd_tcga = pd.read_csv(PATH_LABEL)
dict_sampleid_to_labelnum = dict(zip(pd_tcga['0'], pd_tcga['1']))
for k in list(dict_sampleid_to_labelnum.keys()):
    dict_sampleid_to_labelnum[k.split('/')[1]] = dict_sampleid_to_labelnum[k]
    del dict_sampleid_to_labelnum[k]

# you may shuffle this///
key = list(dict_sampleid_to_labelnum.keys())
if len(key) != len(list(set(key))):
    print(f'multiple sample id exist in TCGA.csv')
NUM_ALL = len(key)
print(f'There are {NUM_ALL} samples in total.')

processed_sample = 0
## Train
for fold in range(1,6):
    path_to_fold = os.path.join(PATH_PKL, 'train', f'fold{fold}')
    # create all necessary directories
    for category in list(MAP.values()):
        Path(os.path.join(path_to_fold, category)).mkdir(parents=True, exist_ok=True)
    for idx_sample in range((fold-1)*NUM_FOLD, (fold*NUM_FOLD)):
        sample_id = key[idx_sample]
        label_str = MAP[dict_sampleid_to_labelnum[sample_id]]

        data_csvpath = os.path.join(PATH_DATA, f'{sample_id}.csv')
        if os.path.isfile(data_csvpath):
            data_np = pd.read_csv(data_csvpath).to_numpy()
            with open(os.path.join(PATH_PKL, 'train', f'fold{fold}', label_str, f'{sample_id}.pkl'), "wb") as fp:
                pickle.dump({'feature': data_np, 'label': label_str}, fp)
        processed_sample += 1
        print(f'{processed_sample} / {NUM_ALL} were processed!!')
    else:
        print(f'{sample_id} does NOT exist')

## Test
path_to_test = os.path.join(PATH_PKL, 'test')
# create all necessary directories
for category in list(MAP.values()):
    Path(os.path.join(path_to_test, category)).mkdir(parents=True, exist_ok=True)
for idx_sample in range(NUM_TR, NUM_ALL):
    sample_id = key[idx_sample]
    label_str = MAP[dict_sampleid_to_labelnum[sample_id]]

    data_csvpath = os.path.join(PATH_DATA, f'{sample_id}.csv')
    if os.path.isfile(data_csvpath):
        data_np = pd.read_csv(data_csvpath).to_numpy()
        with open(os.path.join(path_to_test, label_str, f'{sample_id}.pkl'), "wb") as fp:
            pickle.dump({'feature': data_np, 'label': label_str}, fp)
        processed_sample += 1
        print(f'{processed_sample} / {NUM_ALL} were processed!!')
    else:
        print(f'{sample_id} does NOT exist')
