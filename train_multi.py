import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from typing import Tuple
from utils import adjust_learning_rate, loss, Dataset_pkl, CosineAnnealingWarmUpSingle, CosineAnnealingWarmUpRestarts, optimal_thresh, multi_label_roc, save_checkpoint
import models as milmodels
from tqdm import tqdm, trange
import numpy as np
from utils.sam import SAM
from utils.bypass_bn import enable_running_stats, disable_running_stats
import socket
from datetime import datetime
import torchvision
import math
import threading

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
# /nfs/strange/shared/hazel/stad_simclr_lr1/train
parser = argparse.ArgumentParser(description='MIL Training') 
parser.add_argument('--data-root', default='/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50', help='path to dataset')
parser.add_argument('--fold', default=5, help='number of fold for cross validation')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 1)')
parser.add_argument('--scheduler-centroid', default='single', choices=['None', 'single', 'multi'], type=str, help='loss scheduler')
parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='the total batch size on the current node (DDP)')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

parser.add_argument('--dataset', default='CAMELYON16', choices=['CAMELYON16', 'tcga_lung', 'tcga_stad'], type=str, help='dataset type')
parser.add_argument('--train-instance', default='None', choices=['None', 'semisup1', 'semisup2', 'intrainstance_divdis',
                                                                                'interinstance_vc','interinstance_cosine', 'intrainstance_vc',
                                                                                'intrainstance_cosine'], type=str, help='instance loss type')
parser.add_argument('--ic-num-head', default=5, type=int, help='# of projection head for each instance token')
parser.add_argument('--ic-depth', default=2, choices=[0,1,2], type=int, help='layer number of projection head for instance tokens')
parser.add_argument('--weight-agree', default=1.0, type=float, help='weight for the agree loss, eg, center, cosine')
parser.add_argument('--weight-disagree', default=1.0, type=float, help='weight for the disagree loss, eg, variance loss, contrastive')
parser.add_argument('--weight-cov', default=1.0, type=float, help='weight for the covariance loss')
parser.add_argument('--stddev-disagree', default=1.0, type=float, help='std dev threshold for disagree loss')
parser.add_argument('--optimizer-nc', default='sgd', choices=['sgd', 'adam', 'adamw'], type=str, help='optimizer for negative centroid')
parser.add_argument('--lr', default=0.003, type=float, metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--lr-aux', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lr-center', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--mil-model', default='Dsmil', type=str, help='use pre-training method')

parser.add_argument('--process-num', default=8, type=int, help='number of threads')

parser.add_argument('--pushtoken', default=False, help='Push Bullet token')

def run_fold(args_list, fold, txt_name) -> Tuple:

    random.seed(args_list.seed)
    torch.manual_seed(args_list.seed)
    torch.cuda.manual_seed_all(args_list.seed)
    torch.backends.cudnn.benchmark = True
    # cudnn.deterministic = True

    dataset_train = Dataset_pkl(path_pretrained_pkl_root=args_list.data_root, fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='train', num_classes=args_list.num_classes, seed=args_list.seed)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    args.num_step = len(loader_train)

    dataset_val = Dataset_pkl(path_pretrained_pkl_root=args.data_root, fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='val', num_classes=args_list.num_classes, seed=args_list.seed)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    model = [None]*args.process_num
    file_name = [None]*args.process_num
    auc_best = [0.0]*args.process_num
    epoch_best = [0]*args.process_num
    auc_val = [0.0]*args.process_num
    acc_val = [0.0]*args.process_num
    auc_test = [0.0]*args.process_num
    acc_test = [0.0]*args.process_num
    auc_tr = [0.0]*args.process_num
    acc_tr = [0.0]*args.process_num

    for idx_process in range(args.process_num):
        model[idx_process] = milmodels.__dict__[args.mil_model[idx_process]](args=args, ma_dim_in=2048).cuda()
        file_name[idx_process] = f'{idx_process}_lr{args.lr}_lr_center{args.lr_center}_fold{fold}.pth'

    for epoch in trange(1, (args.epochs+1)):        
        train(loader_train, model)
        auc, acc = validate(loader_val, model, args)
        for idx_process in range(args.process_num):
            if np.mean(auc[idx_process]) > auc_best[idx_process]:
                epoch_best[idx_process] = epoch
                auc_best[idx_process] = np.mean(auc[idx_process])
                auc_val[idx_process] = auc[idx_process]
                acc_val[idx_process] = acc[idx_process]
                torch.save({'state_dict': model[idx_process].state_dict()}, file_name[idx_process])
            print(f'auc val: {auc_val}')
    

    dataset_test = Dataset_pkl(path_pretrained_pkl_root=args.data_root, fold_now=999, fold_all=9999, shuffle_slide=False, shuffle_patch=False, split='test', num_classes=args.num_classes, seed=args.seed)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    for idx_process in range(args.process_num):
        _checkpoint = torch.load(file_name[idx_process], map_location=f'cuda:{args.gpus[idx_process]}')
        model[idx_process].load_state_dict(_checkpoint['state_dict'])
        os.remove(file_name[idx_process])

    auc_test, acc_test = validate(loader_test, model, args)
    auc_tr, acc_tr = validate(loader_train, model, args)

    del dataset_train, loader_train, dataset_val, loader_val
    print(f'fold [{fold}]: epoch_best ==> {epoch_best}')
    print(f'auc_tr: {auc_tr}, auc_val: {auc_val}, auc_test: {auc_test},')
    
    return auc_test, acc_test, auc_val, acc_val, auc_tr, acc_tr, dataset_test.category_idx, epoch_best

def train(train_loader, model_list):
    torch.cuda.empty_cache()
    def train_single(_image, _target, _model, _gpu):
        # images --> #bags x #instances x #dims
        _image = _image.type(torch.FloatTensor).cuda(_gpu, non_blocking=True)
        # target --> #bags x #classes
        _target = _target.type(torch.FloatTensor).cuda(_gpu, non_blocking=True)

        _model.update(_image, _target)

    for _model in model_list:
        _model.train()
    threads = [None]*args.process_num
    for i, (images, target) in enumerate(train_loader):
        for i in range(args.process_num):
            threads[i] = threading.Thread(target = train_single, args=(images, target, model_list[i], args.gpus[i]))
            threads[i].start()
        
        for _thread in threads:
            _thread.join()


def validate(val_loader, model_list, args):
    torch.cuda.empty_cache()

    auc = [None]*args.process_num
    acc = [None]*args.process_num
    bag_labels = []
    bag_predictions = []
    for i in range(args.process_num):
        bag_labels.append([])
        bag_predictions.append([])

    def valid_single(_image, _target, _model, _gpu, _process_idx):
        # images --> #bags x #instances x #dims
        _image = _image.type(torch.FloatTensor).cuda(_gpu, non_blocking=True)

        prob_bag, _ = _model.infer(_image)
        bag_predictions[_process_idx].append(prob_bag.squeeze(0).cpu().numpy())
        bag_labels[_process_idx].append(_target.squeeze(0).numpy())

    for _model in model_list:
        _model.eval()
    threads = [None]*args.process_num
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            for i in range(args.process_num):
                threads[i] = threading.Thread(target = valid_single, args=(images, target, model_list[i], args.gpus[i]))
                threads[i].start()
            
            for _thread in threads:
                _thread.join()

        # bag_labels --> #bag x #classes
        bag_labels = np.array(bag_labels)
        # bag_predictions --> #bag x #classes
        bag_predictions = np.array(bag_predictions)
        assert len(bag_predictions.shape) == 2
        auc, acc = multi_label_roc(bag_labels, bag_predictions, num_classes=bag_labels.shape[-1], pos_label=1)


        for i in range(args.process_num):
            # bag_labels --> #bag x #classes
            bag_labels[i] = np.array(bag_labels[i])
            bag_predictions[i] = np.array(bag_predictions[i])
            assert len(bag_predictions[i].shape) == 2
            auc[i], acc[i] = multi_label_roc(bag_labels, bag_predictions, num_classes=bag_labels.shape[-1], pos_label=1)
    return auc, acc

if __name__ == '__main__':
    args_common=parser.parse_args()
    args_list = [parser.parse_args()]*args_common.process_num

    ################################### 여길 바꿔 ####################################
    _train_instance = ['None']*4
    _ic_num_head = [5]*4
    _ic_depth = [2]*4
    _weight_agree = [1.0]*4
    _weight_disagree = [1.0]*4
    _lr = [0.003]*4
    _lr_center = [0.0001]*4
    _mil_model = ['Dsmil']*4
    _gpus = [0,0,1,1]
    ##################################################################################

    for idx_process in range(args_common.processes):
        # args.num_classes=2 if args.dataset=='tcga_lung' else 1
        args_list[idx_process].num_classes=1
        
        args_list[idx_process].train_instance = _train_instance[idx_process]
        args_list[idx_process].ic_num_head = _ic_num_head[idx_process]
        args_list[idx_process].ic_depth = _ic_depth[idx_process]
        args_list[idx_process].weight_agree = _weight_agree[idx_process]
        args_list[idx_process].weight_disagree = _weight_disagree[idx_process]
        args_list[idx_process].lr = _lr[idx_process]
        args_list[idx_process].lr_center = _lr_center[idx_process]
        args_list[idx_process].mil_model = _mil_model[idx_process]
        args_list[idx_process].gpus = _gpus[idx_process]

        if args_list[idx_process].mil_model == 'Dtfd':
            args_list[idx_process].epochs = 200
        elif args_list[idx_process].mil_model == 'Dsmil':
            args_list[idx_process].epochs = 40
        elif args_list[idx_process].mil_model == 'Attention':
            args_list[idx_process].epochs = 100
        elif args_list[idx_process].mil_model == 'GatedAttention':
            args_list[idx_process].epochs = 100

    t_start = time.time()

    
    txt_name = [0]*args_common.process_num
    acc_fold_tr = [[]]*args_common.process_num
    auc_fold_tr = [[]]*args_common.process_num

    acc_fold_val = [[]]*args_common.process_num
    auc_fold_val = [[]]*args_common.process_num

    acc_fold_test = [[]]*args_common.process_num
    auc_fold_test = [[]]*args_common.process_num

    for idx_process in range(args_common.process_num):
    
        txt_name[idx_process] = f'{datetime.today().strftime("%m%d")}_{args_list[idx_process].dataset}_{args_list[idx_process].mil_model}_scheduler_centroid{args_list[idx_process].scheduler_centroid}_train_instance{args_list[idx_process].train_instance}' +\
        f'_ic_num_head{args_list[idx_process].ic_num_head}_ic_depth{args_list[idx_process].ic_depth}_optimizer_nc{args_list[idx_process].optimizer_nc}' +\
        f'_weight_agree{args_list[idx_process].weight_agree}_weight_disagree{args_list[idx_process].weight_disagree}_weight_cov{args_list[idx_process].weight_cov}_stddev_disagree{args_list[idx_process].stddev_disagree}'

    for fold_num in range(1, args_common.fold+1):
        auc_test, acc_test, auc_val, acc_val, auc_tr, acc_tr, category_idx, epoch_best = run_fold(args_list, fold_num, txt_name)

        acc_fold_tr[idx_process].append(acc_tr[idx_process])
        auc_fold_tr[idx_process].append(auc_tr[idx_process])
        acc_fold_val[idx_process].append(acc_val[idx_process])
        auc_fold_val[idx_process].append(auc_val[idx_process])
        acc_fold_test[idx_process].append(acc_test[idx_process])
        auc_fold_test[idx_process].append(auc_test[idx_process])

    print(f'Training took {round(time.time() - t_start, 3)} seconds')
    print(f'Best epoch: {epoch_best}')
    
    # for fold_num in range(1, args.fold+1):
    # for idx_process in range(args.process_num):
    #     print(f'Fold {fold_num}: ACC TR({acc_fold_tr[fold_num-1]}), AUC TR({auc_fold_tr[fold_num-1]})')
    #     print(f'Fold {fold_num}: ACC VAL({acc_fold_val[fold_num-1]}), AUC VAL({auc_fold_val[fold_num-1]})')
    #     print(f'Fold {fold_num}: ACC TEST({acc_fold_test[fold_num-1]}), AUC TEST({auc_fold_test[fold_num-1]})')
    print(f'{args_common.fold} folds average')
    for idx_process in range(args_common.process_num):
        auc_fold_tr[idx_process] = np.mean(auc_fold_tr[idx_process], axis=0)
        auc_fold_val[idx_process] = np.mean(auc_fold_val[idx_process], axis=0)
        auc_fold_test[idx_process] = np.mean(auc_fold_test[idx_process], axis=0)

        acc_fold_tr[idx_process] = np.mean(acc_fold_tr[idx_process], axis=0)
        acc_fold_val[idx_process] = np.mean(acc_fold_val[idx_process], axis=0)
        acc_fold_test[idx_process] = np.mean(acc_fold_test[idx_process], axis=0)
    
        with open(txt_name[idx_process] + '.txt', 'a' if os.path.isfile(txt_name[idx_process] + '.txt') else 'w') as f:
            f.write(f'===================== LR-mil: {args_list[idx_process].lr[idx_process]} || LR-negative center: {args_list[idx_process].lr_center[idx_process]} =======================\n')
            # if args.num_classes[idx_process] == 1:
            #     f.write(f'AUC TR: {auc_fold_tr[0]}\n')
            #     f.write(f'AUC VAL: {auc_fold_val[0]}\n')
            #     f.write(f'AUC TEST: {auc_fold_test[0]}\n')
            # elif args.num_classes[idx_process] == 2:
            #     for i, k in enumerate(category_idx.keys()):
            #         f.write(f'AUC TR({k}): {auc_fold_tr[i]}\n')
            #         f.write(f'AUC VAL({k}): {auc_fold_val[i]}\n')
            #         f.write(f'AUC TEST({k}): {auc_fold_test[i]}\n')
        
            f.write(f'AUC TR: {auc_fold_tr[idx_process]}\n')
            f.write(f'AUC VAL: {auc_fold_val[idx_process]}\n')
            f.write(f'AUC TEST: {auc_fold_test[idx_process]}\n')
            
            f.write(f'ACC TR: {acc_fold_tr[idx_process]}\n')
            f.write(f'ACC VAL: {acc_fold_val[idx_process]}\n')
            f.write(f'ACC TEST: {acc_fold_test[idx_process]}\n')
            
            # f.write(f'ACC TR: {sum(acc_fold_tr)/float(len(acc_fold_tr))}\n')
            # f.write(f'AUC TR (Average): {np.mean(auc_fold_tr)}\n')
            # f.write(f'ACC VAL: {sum(acc_fold_val)/float(len(acc_fold_val))}\n')
            # f.write(f'AUC VAL (Average): {np.mean(auc_fold_val)}\n')
            # f.write(f'ACC TEST: {sum(acc_fold_test)/float(len(acc_fold_test))}\n')
            # f.write(f'AUC TEST (Average): {np.mean(auc_fold_test)}\n')
            f.write(f'==========================================================================================\n\n\n')
    
    if args_common.pushtoken:
        from pushbullet import API
        import socket
        pb = API()
        pb.set_token(args_common.pushtoken)
        push = pb.send_note('MIL train finished', f'{socket.gethostname()}')