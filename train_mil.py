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
from utils import adjust_learning_rate, loss, Dataset_pkl, CosineAnnealingWarmUpSingle, optimal_thresh, five_scores, multi_label_roc, save_checkpoint
import models as milmodels
from tqdm import tqdm, trange
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
# /nfs/strange/shared/hazel/stad_simclr_lr1/train
parser = argparse.ArgumentParser(description='MIL Training')
parser.add_argument('--pkl-root', default='/nfs/strange/shared/hazel/stad_cv', help='path to pkls')
parser.add_argument('--data-root', default='/nfs/strange/shared/hazel/stad_simclr_lr1', help='path to dataset')
parser.add_argument('--dataset', default='CAMELYON16', choices=['CAMELYON16', 'tcga_lung', 'tcga_stad'], type=str, help='dataset type')
parser.add_argument('--weight', default='simclr_lr1', help='weight folder')
parser.add_argument('--fold', default=5, help='number of fold for cross validation')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str, help='optimizer')
parser.add_argument('--loss', default='bce', choices=['bce'], type=str, help='loss function')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='the total batch size on the current node (DDP)')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
# DTFD
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--mil-model', default='MilBase', choices=['MilBase'], type=str, help='use pre-training method')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

def run_fold(args, fold) -> Tuple:

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # cudnn.deterministic = True
    device = 0
    num_classes=2 if args.dataset=='tcga_lung' else 1

    model = milmodels.__dict__[args.mil_model](dim_out=num_classes).cuda() 

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss().cuda()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 0, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.sgd(model.parameters(), 0, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Undefined Optimizer!!')
    
    # dataset_train = Dataset_pkl(path_pkl_root=os.path.join(args.data_root, args.dataset, f'fold{fold}.pkl'), fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, if_train=True, seed=args.seed)
    # dataset_train = Dataset_pkl(path_fold_pkl=os.path.join(args.pkl_root), path_pretrained_pkl_root='/nfs/strange/shared/hazel/stad_simclr_lr1', fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='train', num_classes=num_classes, seed=args.seed)
    dataset_train = Dataset_pkl(path_fold_pkl=os.path.join(args.pkl_root), path_pretrained_pkl_root=args.data_root, fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='train', num_classes=num_classes, seed=args.seed)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    dataset_val = Dataset_pkl(path_fold_pkl=os.path.join(args.pkl_root), path_pretrained_pkl_root=args.data_root, fold_now=fold, fold_all=args.fold, shuffle_slide=False, shuffle_patch=False, split='val', num_classes=num_classes, seed=args.seed)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 고쳐야 함
    scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loader_train))
    scaler = torch.cuda.amp.GradScaler()

    for epoch in trange(1, args.epochs):        
        train(loader_train, model, criterion, optimizer, device, scheduler, scaler)
    acc, auc = validate(loader_val, model, criterion, device, args)
    
    return acc, auc

def train(train_loader, model, criterion, optimizer, device, scheduler, scaler):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # compute output
            output = model(images)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

def validate(val_loader, model, criterion, device, args):
    bag_labels = []
    bag_predictions = []
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            bag_labels.append(target.squeeze(0).numpy())
            images = images.cuda(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                # compute output
                output = model(images)
                # loss = criterion(output, target.cuda(device, non_blocking=True))
            # bag_predictions.append(torch.sigmoid(output).cpu().squeeze().numpy())
            bag_predictions.append(torch.sigmoid(output.type(torch.DoubleTensor)).squeeze(0).cpu().numpy())
        
        if args.dataset == 'tcga_lung':
            bag_labels = np.array(bag_labels)
            bag_predictions = np.array(bag_predictions)
            auc, _, thresholds_optimal = multi_label_roc(bag_labels, bag_predictions, num_classes=2, pos_label=1)
            acc = float(torch.tensor(bag_labels).eq(torch.tensor(bag_predictions)).sum().item())/float(torch.tensor(bag_labels).numel())
        else:
            acc, auc, precision, recall, fscore = five_scores(bag_labels, bag_predictions)

    return acc, auc

if __name__ == '__main__':
    args = parser.parse_args()
    accs = []
    aucs = []
    t_start = time.time()
    for fold_num in range(1, args.fold+1):
        _acc, _auc = run_fold(args, fold_num)
        accs.append(_acc)
        aucs.append(_auc)

    print(f'Training took {round(time.time() - t_start, 3)} seconds')
    
    for fold_num in range(1, args.fold+1):
        print(f'Fold {fold_num}: ACC({accs[fold_num-1]}), AUC({aucs[fold_num-1]})')