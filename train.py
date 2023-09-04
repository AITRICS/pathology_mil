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

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
# /nfs/strange/shared/hazel/stad_simclr_lr1/train
parser = argparse.ArgumentParser(description='MIL Training') 
parser.add_argument('--data-root', default='/home/chris/storage/camelyon16_eosin_224_16_pkl_0524/swav_res50', help='path to dataset')
# parser.add_argument('--data-root', default='/mnt/aitrics_ext/ext01/shared/tcgalung_dsmil', help='path to dataset')
parser.add_argument('--fold', default=5, help='number of fold for cross validation')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 1)')
parser.add_argument('--scheduler-centroid', default='single', choices=['None', 'single', 'multi'], type=str, help='loss scheduler')
parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='the total batch size on the current node (DDP)')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

parser.add_argument('--dataset', default='CAMELYON16', choices=['CAMELYON16', 'tcga_lung', 'tcga_stad'], type=str, help='dataset type')
# parser.add_argument('--dataset', default='tcga_lung', choices=['CAMELYON16', 'tcga_lung', 'tcga_stad'], type=str, help='dataset type')
parser.add_argument('--train-instance', default='interinstance_vic', choices=['None', 'semisup1', 'semisup2', 'divdis',
                                                                                'interinstance_vi','interinstance_vic', 'intrainstance_vc',
                                                                                'intrainstance_cosine'], type=str, help='instance loss type')
parser.add_argument('--ic-num-head', default=1, type=int, help='# of projection head for each instance token')
parser.add_argument('--ic-depth', default=1, choices=[0,1,2,3,4], type=int, help='layer number of projection head for instance tokens')
parser.add_argument('--weight-agree', default=1.0, type=float, help='weight for the agree loss, eg, center, cosine')
parser.add_argument('--weight-disagree', default=0.3, type=float, help='weight for the disagree loss, eg, variance loss, contrastive')
parser.add_argument('--weight-cov', default=1.0, type=float, help='weight for the covariance loss')
parser.add_argument('--stddev-disagree', default=1.5, type=float, help='std dev threshold for disagree loss')
parser.add_argument('--optimizer-nc', default='adamw', choices=['sgd', 'adam', 'adamw'], type=str, help='optimizer for negative centroid')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--lr-aux', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lr-center', default=0.00001, type=float, help='initial learning rate')
parser.add_argument('--mil-model', default='Attention', type=str, help='use pre-training method')
parser.add_argument('--passing-v', default=1, choices=[0,1], type=int, help='passing_v for dsmil')
parser.add_argument('--dsmil-method', default='BClassifier_ascend', choices=['BClassifier_basic', 'BClassifier_ascend'], type=str, help='BCLassifier type for dsmil')

parser.add_argument('--pushtoken', default=False, help='Push Bullet token')

def run_fold(args, fold, txt_name) -> Tuple:
    _std_neg=[]
    _std_pos=[]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # cudnn.deterministic = True

    dataset_train = Dataset_pkl(path_pretrained_pkl_root=args.data_root, fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='train', num_classes=args.num_classes, seed=args.seed)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    args.num_step = len(loader_train)
    if args.dataset == 'CAMELYON16':
        args.num_step_neg = args.num_neg[fold-1]
    elif args.dataset == 'tcga_lung':
        args.num_step_neg = args.num_neg[fold-1]

    dataset_val = Dataset_pkl(path_pretrained_pkl_root=args.data_root, fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='val', num_classes=args.num_classes, seed=args.seed)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
      
    model = milmodels.__dict__[args.mil_model](args=args, ma_dim_in=512 if args.dataset == 'tcga_lung' else 2048).cuda()
    
    auc_best = 0.0
    epoch_best = 0
    file_name = f'{txt_name}_lr{args.lr}_lr_center{args.lr_center}_fold{fold}.pth'
    for epoch in trange(1, (args.epochs+1)):        
        train(loader_train, model)
        auc, acc = validate(loader_val, model, args, _std_neg, _std_pos)
        if np.mean(auc) > auc_best:
            epoch_best = epoch
            auc_best = np.mean(auc)
            auc_val = auc
            acc_val = acc
            torch.save({
                'state_dict': model.state_dict(),
                'negative_centroid': model.negative_centroid if hasattr(model, 'negative_centroid') else None,
                'negative_std': model.negative_std if hasattr(model, 'negative_std') else None,
                }, file_name)
        print(f'auc val: {auc}')
    
    print(f'[FIN] _std_neg (VAL): {_std_neg}')
    print(f'[FIN] _std_pos (VAL): {_std_pos}')
    dataset_test = Dataset_pkl(path_pretrained_pkl_root=args.data_root, fold_now=999, fold_all=9999, shuffle_slide=False, shuffle_patch=False, split='test', num_classes=args.num_classes, seed=args.seed)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    checkpoint = torch.load(file_name, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])
    os.remove(file_name)

    _std_neg=[]
    _std_pos=[]
    auc_test, acc_test = validate(loader_test, model, args, _std_neg, _std_pos)
    print(f'[FIN] _std_neg (TEST): {_std_neg}')
    print(f'[FIN] _std_pos (TEST): {_std_pos}')
    _std_neg=[]
    _std_pos=[]
    auc_tr, acc_tr = validate(loader_train, model, args, _std_neg, _std_pos)
    print(f'[FIN] _std_neg (TR): {_std_neg}')
    print(f'[FIN] _std_pos (TR): {_std_pos}')

    del dataset_train, loader_train, dataset_val, loader_val
    print(f'fold [{fold}]: epoch_best ==> {epoch_best}')
    print(f'auc_tr: {auc_tr}, auc_val: {auc_val}, auc_test: {auc_test},')
    
    return auc_test, acc_test, auc_val, acc_val, auc_tr, acc_tr, dataset_test.category_idx, epoch_best

def train(train_loader, model):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        # images --> #bags x #instances x #dims
        images = images.type(torch.FloatTensor).cuda(args.device, non_blocking=True)
        # target --> #bags x #classes
        target = target.type(torch.FloatTensor).cuda(args.device, non_blocking=True)

        model.update(images, target)

def validate(val_loader, model, args, _std_neg, _std_pos):
    bag_labels = []
    bag_predictions = []
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            # target --> #bags x #classes
            # bag_labels --> #classes
            bag_labels.append(target.squeeze(0).numpy())
            # images --> #bags x #instances x #dims
            images = images.type(torch.FloatTensor).cuda(args.device, non_blocking=True)
            
            # output --> #bags x #classes
            prob_bag, _ = model.infer(images, bag_labels[-1])
            #classes  (prob)
            bag_predictions.append(prob_bag.squeeze(0).cpu().numpy())

        if len(model.std_neg) > 0:
            _std_neg.append(sum(model.std_neg)/float(len(model.std_neg)))
            _std_pos.append(sum(model.std_pos)/float(len(model.std_pos)))

        # bag_labels --> #bag x #classes
        bag_labels = np.array(bag_labels)
        # bag_predictions --> #bag x #classes
        bag_predictions = np.array(bag_predictions)
        assert len(bag_predictions.shape) == 2
        auc, acc = multi_label_roc(bag_labels, bag_predictions, num_classes=bag_labels.shape[-1], pos_label=1)

    return auc, acc

if __name__ == '__main__':
    args = parser.parse_args()
    
    args.pretrain_type = args.data_root.split("/")[-2:]
    # txt_name = f'{args.dataset}_{args.pretrain_type}_downstreamLR_{args.lr}_optimizer_{args.optimizer}_epoch{args.epochs}_wd{args.weight_decay}'    
    txt_name = f'{datetime.today().strftime("%m%d")}_{args.dataset}_{args.mil_model}_train_instance{args.train_instance}_weight_cov{args.weight_cov}' +\
    f'_weight_agree{args.weight_agree}_weight_disagree{args.weight_disagree}_stddev_disagree{args.stddev_disagree}_passing_v{args.passing_v}_dsmil_method{args.dsmil_method}'
    acc_fold_tr = []
    auc_fold_tr = []

    acc_fold_val = []
    auc_fold_val = []

    acc_fold_test = []
    auc_fold_test = []

    args.num_classes=2 if args.dataset=='tcga_lung' else 1
    # args.num_classes=1
    # args.output_bag_dim=2
    args.device = 0

    if args.dataset == 'CAMELYON16':
        args.num_neg = [[159], [159], [160], [159], [159]]
    elif args.dataset == 'tcga_lung':
        # args.num_neg = [[82,85], [82,85], [82,85], [81,86], [81,86]]
        args.num_neg = [[326, 342], [326, 342], [326, 342], [327, 341], [327, 341]]

    if args.mil_model == 'Dtfd':
        args.epochs = 200
    elif args.mil_model == 'Dsmil':
        args.epochs = 40
    elif args.mil_model == 'Attention':
        args.epochs = 100
    elif args.mil_model == 'GatedAttention':
        args.epochs = 100

    t_start = time.time()
    for fold_num in range(1, args.fold+1):
        auc_test, acc_test, auc_val, acc_val, auc_tr, acc_tr, category_idx, epoch_best = run_fold(args, fold_num, txt_name)

        acc_fold_tr.append(acc_tr)
        auc_fold_tr.append(auc_tr)
        acc_fold_val.append(acc_val)
        auc_fold_val.append(auc_val)
        acc_fold_test.append(acc_test)
        auc_fold_test.append(auc_test)

    print(f'Training took {round(time.time() - t_start, 3)} seconds')
    print(f'Best epoch: {epoch_best}')
    
    for fold_num in range(1, args.fold+1):
        print(f'Fold {fold_num}: ACC TR({acc_fold_tr[fold_num-1]}), AUC TR({auc_fold_tr[fold_num-1]})')
        print(f'Fold {fold_num}: ACC VAL({acc_fold_val[fold_num-1]}), AUC VAL({auc_fold_val[fold_num-1]})')
        print(f'Fold {fold_num}: ACC TEST({acc_fold_test[fold_num-1]}), AUC TEST({auc_fold_test[fold_num-1]})')
    print(f'{args.fold} folds average')
    auc_fold_tr = np.mean(auc_fold_tr, axis=0)
    auc_fold_val = np.mean(auc_fold_val, axis=0)
    auc_fold_test = np.mean(auc_fold_test, axis=0)
    
    with open(txt_name + '.txt', 'a' if os.path.isfile(txt_name + '.txt') else 'w') as f:
        f.write(f'===================== LR-mil: {args.lr} || LR-negative center: {args.lr_center} =======================\n')
        if args.num_classes == 1:
            f.write(f'AUC TR: {auc_fold_tr[0]}\n')
            f.write(f'AUC VAL: {auc_fold_val[0]}\n')
            f.write(f'AUC TEST: {auc_fold_test[0]}\n')
        elif args.num_classes == 2:
            for i, k in enumerate(category_idx.keys()):
                f.write(f'AUC TR({k}): {auc_fold_tr[i]}\n')
                f.write(f'AUC VAL({k}): {auc_fold_val[i]}\n')
                f.write(f'AUC TEST({k}): {auc_fold_test[i]}\n')
        f.write(f'ACC TR: {sum(acc_fold_tr)/float(len(acc_fold_tr))}\n')
        f.write(f'AUC TR (Average): {np.mean(auc_fold_tr)}\n')
        f.write(f'ACC VAL: {sum(acc_fold_val)/float(len(acc_fold_val))}\n')
        f.write(f'AUC VAL (Average): {np.mean(auc_fold_val)}\n')
        f.write(f'ACC TEST: {sum(acc_fold_test)/float(len(acc_fold_test))}\n')
        f.write(f'AUC TEST (Average): {np.mean(auc_fold_test)}\n')
        f.write(f'==========================================================================================\n\n\n')
    
    if args.pushtoken:
        from pushbullet import API
        import socket
        pb = API()
        pb.set_token(args.pushtoken)
        push = pb.send_note('MIL train finished', f'{socket.gethostname()}')