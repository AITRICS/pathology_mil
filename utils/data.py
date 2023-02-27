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
    
    def __init__(self, path_fold_pkl: str, path_pretrained_pkl_root: str, fold_now:int, fold_all:int=5, shuffle_slide: bool=True, shuffle_patch: bool=True, split: str='train', num_classes: int=1, seed:int=1):
        """
        path_fold_pkl: path before fold{FOLD_NUMBER}.pkl
        path_pretrained_pkl_root: path before train/test
        """
        super().__init__()        
        assert (fold_now > 0) and (fold_now <= fold_all)
        self.path_pretrained_pkl_root = path_pretrained_pkl_root
        self.shuffle_patch = shuffle_patch
        self.split = split
        self.seed = seed

        self.rd = random.Random(seed)

        #  Normal must be 0
        print(f'=== Category Indexing ===')
        self.category_idx = {}
        if 'normal' in os.listdir(os.path.join(path_pretrained_pkl_root, 'train' if (split=='train' or split=='val') else 'test')):
            assert num_classes == 1
            self.category_idx['normal'] = np.zeros(1)
            _list_category = os.listdir(os.path.join(path_pretrained_pkl_root, 'train' if (split=='train' or split=='val') else 'test'))
            _list_category.remove('normal')
            assert len(_list_category) == 1
            self.category_idx[_list_category[0]] = np.ones(1)
            
        else:
            _categories = os.listdir(os.path.join(path_pretrained_pkl_root, 'train' if (split=='train' or split=='val') else 'test'))
            _categories.sort()
            for i, _category in enumerate(_categories):
                if num_classes > 1:
                    _temp = np.zeros(num_classes)
                    _temp[i] = 1
                    self.category_idx[_category] = _temp
                else:
                    _temp = np.zeros(1)
                    _temp[0] = i
                    self.category_idx[_category] = _temp
                    # self.category_idx[_category] = i
        for k, v in self.category_idx.items():
            print(f'{k} ===> {v}')
        print(f'=========================')

        folds = list(range(1, fold_all+1))
        # patient_list = self.rd.shuffle(patient_list)

        if split == 'train':
            self.path_pretrained_pkl = os.path.join(path_pretrained_pkl_root, 'train')
            self.path_pkl = []
            folds.pop(fold_now-1)
            for i in folds:
                self.path_pkl.extend(pickle.load(open(os.path.join(path_fold_pkl, f'fold{i}.pkl'), 'rb')))
        elif split == 'val':
            self.path_pretrained_pkl = os.path.join(path_pretrained_pkl_root, 'train')
            self.path_pkl = pickle.load(open(os.path.join(path_fold_pkl, f'fold{fold_now}.pkl'), 'rb'))
        elif split == 'test':
            # self.path_pretrained_pkl = os.path.join(path_pretrained_pkl_root, 'test')
            self.path_pkl = glob.glob(os.path.join(path_pretrained_pkl_root, f'test','*.pkl'))
            self.path_pkl.sort()
        
        if shuffle_slide:
            self.rd.shuffle(self.path_pkl)        

    def __len__(self):
        return len(self.path_pkl)

    def __getitem__(self, idx):
        if (self.split == 'train' or self.split == 'val'):
            _data = pickle.load(open(os.path.join(self.path_pretrained_pkl, f'{self.path_pkl[idx]}.pkl'), 'rb'))
        elif self.split == 'test':
            _data = pickle.load(open(self.path_pkl[idx], 'rb'))

        _data_temp = [feat['feature'] for feat in _data['features']]
        if self.shuffle_patch:
            self.rd.shuffle(_data_temp)
        return torch.from_numpy(np.stack(_data_temp, axis=0)), self.category_idx[_data['label']]
        
class rnndata(Dataset):
    def __init__(self, train_list):
        self.train_list = train_list 
        
    def __len__(self): 
        return len(self.train_list)  
    
    def __getitem__(self, idx):
        return self.train_list[idx]['feats'],  self.train_list[idx]['target']
        
        
        
# if __name__ == '__main__':
#     Dataset_pkl(path_list_pkls = )
