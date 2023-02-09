from torch.utils.data import Dataset
import os
import torch
import glob
import random
from typing import List
from torchvision.datasets import ImageFolder
import pickle
import numpy as np
import glob

class Dataset_pkl(Dataset):
    
    def __init__(self, path_pkl_root: str, fold_now:int, fold_all:int=5, shuffle_slide: bool=True, shuffle_patch: bool=True, if_train: bool=True, seed:int=1):
        super().__init__()        
        assert (fold_now > 0) and (fold_now <= fold_all)
        self.shuffle_patch = shuffle_patch
        self.seed = seed

        self.rd = random.Random(seed)
        self.path_pkl = []

        print(f'=== Category Indexing ===')
        self.category_idx = {}
        for i, _category in enumerate(os.listdir(os.path.join(path_pkl_root, 'fold1'))):
            self.category_idx[_category] = i
            print(f'{_category} ===> {i}')
        print(f'=========================')

        folds = list(range(1, fold_all+1))
        # patient_list = self.rd.shuffle(patient_list)

        if if_train:
            folds.pop(fold_now-1)
            for i in folds:
                self.path_pkl.extend(glob.glob(os.path.join(path_root, f'fold{i}','*.pkl')))
        else:
            self.path_pkl = glob.glob(os.path.join(path_root, f'fold{fold_now}','*.pkl'))
        
        if shuffle_slide:
            self.path_pkl = self.rd.shuffle(self.path_pkl)        

    def __len__(self):
        return len(self.path_pkl)

    def __getitem__(self, idx):
        _data = pickle.load(open(self.path_pkl(idx), 'rb'))
        _data_temp = [feat['feature'] for feat in _data['features']]
        if self.shuffle_patch:
            _data_temp = self.rd.shuffle(_data_temp)
        return torch.from_numpy(np.stack(_data_temp, axis=0)), self.category_idx[_data['label']]
        


# if __name__ == '__main__':
#     Dataset_pkl(path_list_pkls = )
