from torch.utils.data import Dataset
import os
import torch
import glob
import random
from typing import List

class Dataset_pkl(Dataset):
        
    def __init__(self, path_list_pkls: List, fold_now:int, fold_all:int=5, seed:int=1):
        super().__init__()        
        assert (fold_now > 0) and (fold_now <= fold_all)

        path_list_pkls = path_list_pkls.sort()
        patient_list = list(set(map(path_list_pkls, lambda a: a.split('-dx')[0])))

        rd = random.Random(seed)
        patient_list = rd.shuffle(patient_list)

        self.path_pkl
        

    def __len__(self):
        return len(self.path_pkl)

    def __getitem__(self, idx):
        return self.path_pkl(idx)
        