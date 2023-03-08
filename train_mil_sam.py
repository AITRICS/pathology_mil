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

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
# /nfs/strange/shared/hazel/stad_simclr_lr1/train
parser = argparse.ArgumentParser(description='MIL Training')
parser.add_argument('--data-root', default='/mnt/aitrics_ext/ext01/shared/pathology_mil', help='path to dataset')
parser.add_argument('--fold', default=5, help='number of fold for cross validation')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 1)')
parser.add_argument('--scheduler', default='single', choices=['single', 'multi'], type=str, help='loss scheduler')
parser.add_argument('--loss', default='bce', choices=['bce'], type=str, help='loss function')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='the total batch size on the current node (DDP)')
parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

parser.add_argument('--dataset', default='tcga_stad', choices=['CAMELYON16', 'tcga_lung', 'tcga_stad'], type=str, help='dataset type')
parser.add_argument('--pretrain-type', default='simclr_lr1', help='weight folder')
parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'], type=str, help='optimizer')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
# DTFD: 1e-4, TransMIL: 1e-5
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--mil-model', default='GatedAttention', choices=[ 'milmax', 'milmean', 'Attention', 'GatedAttention'], type=str, help='use pre-training method')


parser.add_argument('--pushtoken', default=False, help='Push Bullet token')

def run_fold(args, fold) -> Tuple:

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # cudnn.deterministic = True

    model = milmodels.__dict__[args.mil_model](dim_in=2048, dim_latent=512, dim_out=args.num_classes).cuda()
    
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss().cuda()
# ['adam', 'sgd', 'adamw', 'swa', 'sam']
    if args.optimizer == 'adam':
        optimizer_based = torch.optim.Adam
    elif args.optimizer == 'sgd':
        optimizer_based = torch.optim.SGD
    elif args.optimizer == 'adamw':
        optimizer_based = torch.optim.AdamW
    # elif args.optim_wrapper == 'sam':
    # https://github.com/davda54/sam
    optimizer = SAM(model.parameters(), optimizer_based, lr=args.lr, weight_decay=args.weight_decay, adaptive=True)
    
    dataset_train = Dataset_pkl(path_fold_pkl=os.path.join(args.data_root, 'cv', args.dataset), path_pretrained_pkl_root=os.path.join(args.data_root, 'features', args.dataset, args.pretrain_type), fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='train', num_classes=args.num_classes, seed=args.seed)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    dataset_val = Dataset_pkl(path_fold_pkl=os.path.join(args.data_root, 'cv', args.dataset), path_pretrained_pkl_root=os.path.join(args.data_root, 'features', args.dataset, args.pretrain_type), fold_now=fold, fold_all=args.fold, shuffle_slide=False, shuffle_patch=False, split='val', num_classes=args.num_classes, seed=args.seed)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 고쳐야 하나..?
    if args.scheduler == 'single':
        scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loader_train))
    elif args.scheduler == 'multi':
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, eta_max=args.lr, step_total=args.epochs * len(loader_train))
    scaler = torch.cuda.amp.GradScaler()

    for epoch in trange(1, (args.epochs+1)):        
        train(loader_train, model, criterion, optimizer, scheduler, scaler)
    auc, acc = validate(loader_val, model, criterion, args)
    
    return auc, acc, dataset_val.category_idx

def train(train_loader, model, criterion, optimizer, scheduler, scaler):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        # images --> #bags x #instances x #dims
        images = images.type(torch.FloatTensor).to(args.device, non_blocking=True)
        # target --> #bags x #classes
        target = target.type(torch.FloatTensor).to(args.device, non_blocking=True)

        # First step
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
            # output --> #bags x #classes
        # enable_running_stats(model)
        # output = model(images)
        # loss = criterion(output, target)
        loss = model.calculate_objective(images, target)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # Second step
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
            # output --> #bags x #classes
        disable_running_stats(model)
        # output = model(images)
        # loss = criterion(output, target)
        loss = model.calculate_objective(images, target)
        loss.backward()
        optimizer.second_step(zero_grad=True)

        scheduler.step()

def validate(val_loader, model, criterion, args):
    bag_labels = []
    bag_predictions = []
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            # target --> #bags x #classes
            # bag_labels --> #classes
            bag_labels.append(target.squeeze(0).numpy())
            # images --> #bags x #instances x #dims
            images = images.cuda(args.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                # output --> #bags X #classes
                logit_bag, _ = model(images)
            #classes  (prob)
            bag_predictions.append(torch.sigmoid(logit_bag.type(torch.DoubleTensor)).squeeze(0).cpu().numpy())

        # bag_labels --> #classes
        bag_labels = np.array(bag_labels)
        # bag_predictions --> #classes
        bag_predictions = np.array(bag_predictions)
        assert len(bag_predictions.shape) == 2
        auc, acc = multi_label_roc(bag_labels, bag_predictions, num_classes=bag_labels.shape[-1], pos_label=1)

    return auc, acc

if __name__ == '__main__':
    args = parser.parse_args()
    # txt_name = f'{args.dataset}_{args.pretrain_type}_downstreamLR_{args.lr}_optimizer_{args.optimizer}_epoch{args.epochs}_wd{args.weight_decay}'
    txt_name = f'{datetime.today().strftime("%m%d")}_{args.dataset}_{args.pretrain_type}_mil_model_{args.mil_model}_epoch{args.epochs}_wd{args.weight_decay}_scheduler_{args.scheduler}'

    acc_fold = []
    auc_fold = []

    args.num_classes=2 if args.dataset=='tcga_lung' else 1
    args.device = 0

    t_start = time.time()
    for fold_num in range(1, args.fold+1):
        _auc, _acc, category_idx = run_fold(args, fold_num)
        acc_fold.append(_acc)
        auc_fold.append(_auc)

    print(f'Training took {round(time.time() - t_start, 3)} seconds')
    
    for fold_num in range(1, args.fold+1):
        print(f'Fold {fold_num}: ACC({acc_fold[fold_num-1]}), AUC({auc_fold[fold_num-1]})')
    print(f'{args.fold} folds average')
    auc_fold = np.mean(auc_fold, axis=0)
    
    with open(txt_name + '.txt', 'a' if os.path.isfile(txt_name + '.txt') else 'w') as f:
        f.write(f'===================== LR-pretrain: {args.pretrain_type} || LR-down: {args.lr} || Optimizer: {args.optimizer}+SAM || scheduler: {args.scheduler} =======================\n')
        if args.num_classes == 1:
            f.write(f'AUC: {auc_fold[0]}\n')
        elif args.num_classes == 2:
            for i, k in enumerate(category_idx.keys()):
                f.write(f'AUC ({k}): {auc_fold[i]}\n')
        f.write(f'ACC: {sum(acc_fold)/float(len(acc_fold))}\n')
        f.write(f'==========================================================================================\n\n\n')
    
    if args.pushtoken:
        from pushbullet import Pushbullet
        import socket
        pb = Pushbullet(args.pushtoken)
        push = pb.push_note('MIL train finished', f'{socket.gethostname()}')