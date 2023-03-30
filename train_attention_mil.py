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
from utils import adjust_learning_rate, loss, Dataset_pkl2, CosineAnnealingWarmUpSingle, CosineAnnealingWarmUpRestarts, optimal_thresh, multi_label_roc, save_checkpoint
import models as milmodels
from tqdm import tqdm, trange
import numpy as np
import socket
from datetime import datetime
import torchvision
import torch.nn.functional as F
import math

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
parser.add_argument('--save', default=False, type=bool)

parser.add_argument('--dataset', default='tcga_stad', choices=['CAMELYON16', 'tcga_lung', 'tcga_stad'], type=str, help='dataset type')
parser.add_argument('--pretrain-type', default='ImageNet_Res50_im', help='weight folder')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--optimizer', default='adamw', choices=['sgd', 'adam', 'adamw'], type=str, help='optimizer')
parser.add_argument('--lr', default=0.000001, type=float, metavar='LR', help='initial learning rate', dest='lr')
# DTFD: 1e-4, TransMIL: 1e-5
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--mil-model', default='MilTransformer', choices=['MilTransformer'], type=str, help='use pre-training method')
parser.add_argument('--pseudo-prob-threshold', default=0.9, type=float, help='instance with probability greater than pseudo_prob_threshold will have pseudo label')
parser.add_argument('--semi-start-epoch', default=20, type=int, help='epoch that starts instance training')
parser.add_argument('--share-proj', action='store_true', help='share projection layer between instance tokens and bag token')
parser.add_argument('--num-layers', default=3, type=int, help='number of transformer layers')

parser.add_argument('--pushtoken', default=False, help='Push Bullet token')

def run_fold(args, fold) -> Tuple:

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # cudnn.deterministic = True

    if 'Res18' in args.pretrain_type:
        dim_in = 512
    else:
        dim_in = 2048

    model = milmodels.__dict__[args.mil_model](dim_in=dim_in, dim_out=args.num_classes, num_layers=args.num_layers, share_proj=args.share_proj).cuda() 

    if args.loss == 'bce':
        # default: reduction: str = 'mean'
        criterion_mean = nn.BCEWithLogitsLoss().cuda()
        criterion_none = nn.BCEWithLogitsLoss(reduction='none').cuda()
    if args.num_classes > 1:
        func_prob = torch.nn.Softmax().cuda()
    elif args.num_classes == 1:
        func_prob = torch.nn.Sigmoid().cuda()

# ['adam', 'sgd', 'adamw', 'swa', 'sam']
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 0, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 0, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), 0, weight_decay=args.weight_decay)
  
    dataset_train = Dataset_pkl2(path_fold_pkl=os.path.join(args.data_root, 'cv', args.dataset), path_pretrained_pkl_root=os.path.join(args.data_root, 'features', args.dataset, args.pretrain_type), fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='train', num_classes=args.num_classes, seed=args.seed)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    dataset_val = Dataset_pkl2(path_fold_pkl=os.path.join(args.data_root, 'cv', args.dataset), path_pretrained_pkl_root=os.path.join(args.data_root, 'features', args.dataset, args.pretrain_type), fold_now=fold, fold_all=args.fold, shuffle_slide=False, shuffle_patch=False, split='val', num_classes=args.num_classes, seed=args.seed)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 고쳐야 하나..?
    if args.scheduler == 'single':
        scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loader_train))
    elif args.scheduler == 'multi':
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, eta_max=args.lr, step_total=args.epochs * len(loader_train))
    scaler = torch.cuda.amp.GradScaler()

    auc_best = 0.0
    epoch_best = 0
    file_name = f'{txt_name}_lr{args.lr}_op_{args.optimizer}_fold{fold}.pth'    
    for epoch in trange(1, args.epochs+1):        
        train(loader_train, model, criterion_mean, criterion_none, optimizer, scheduler, scaler, func_prob, epoch)
        auc, acc = validate(loader_val, model, args)
    
    if np.mean(auc) > auc_best:
            epoch_best = epoch
            auc_best = np.mean(auc)
            auc_val = auc
            acc_val = acc
            torch.save({'state_dict': model.state_dict()}, file_name)

    dataset_test = Dataset_pkl2(path_fold_pkl='hello my name is test', path_pretrained_pkl_root=os.path.join(args.data_root, 'features', args.dataset, args.pretrain_type), fold_now=999, fold_all=9999, shuffle_slide=False, shuffle_patch=False, split='test', num_classes=args.num_classes, seed=args.seed)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    checkpoint = torch.load(file_name, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])
    os.remove(file_name)

    auc_test, acc_test = validate(loader_test, model, args)
    auc_tr, acc_tr = validate(loader_train, model, args)

    del dataset_train, loader_train, dataset_val, loader_val, optimizer, scheduler
    
    return auc_test, acc_test, auc_val, acc_val, auc_tr, acc_tr, dataset_test.category_idx, epoch_best


def train(train_loader, model, criterion_mean, criterion_none, optimizer, scheduler, scaler, func_prob, epoch):
    cnt_positive_1=0
    cnt_over_thresh_1=0
    cnt_positive_2=0
    cnt_over_thresh_2=0
    def train_single(model, images, target, epoch, cnt_positive, cnt_over_thresh, criterion_mean, criterion_none):
        # logit_bag --> #bags x #classes
        # logit_instance --> #bags x #patches x #classes
        logit_bag, logit_instance = model(images)
        # logit_bag += (39./252.)
        # logit_instance += (39./252.)
        # loss_bag = criterion_mean(logit_bag.squeeze(1), target)

        # loss_bag = torchvision.ops.sigmoid_focal_loss(logit_bag+math.log(39./252.), target, reduction='mean')
        loss_bag = criterion_mean(logit_bag+math.log(39./252.), target)

        if epoch >= args.semi_start_epoch:
        # if True:
            # lung 이면 코드 수정해야함. #bags=1 일때만 돌아감!!!
            if target == 0:
                # loss_instance = torchvision.ops.sigmoid_focal_loss(logit_instance+math.log(39./252.), target.unsqueeze(1).repeat(logit_instance.size(0), logit_instance.size(1), logit_instance.size(2)), reduction='mean')
                loss_instance = criterion_mean(logit_instance+math.log(39./252.), target.unsqueeze(1).repeat(logit_instance.size(0), logit_instance.size(1), logit_instance.size(2)))
            else:
                cnt_positive+=1
                #slide x #patches x num_class
                # logit_instance = logit_instance.squeeze(0)
                # #patches x num_class
                prob_instance = func_prob(logit_instance)
                pseudo_label_positive = torch.zeros_like(prob_instance).to(args.device, non_blocking=True)
                pseudo_label_positive[prob_instance>args.pseudo_prob_threshold] = 1.0
                
                mask = pseudo_label_positive.detach().clone()
                mask[prob_instance<(1.0-args.pseudo_prob_threshold)]=1.0
                if (torch.sum(pseudo_label_positive) < 1):
                    # loss_instance = torchvision.ops.sigmoid_focal_loss(torch.max(logit_instance)+math.log(39./252.), torch.ones([], device=args.device))
                    loss_instance = criterion_mean(torch.max(logit_instance)+math.log(39./252.), torch.ones([], device=args.device))
                else:
                    # loss_instance = torch.sum(mask * torchvision.ops.sigmoid_focal_loss(logit_instance+math.log(39./252.), pseudo_label_positive))/torch.sum(mask)
                    loss_instance = torch.sum(mask * criterion_none(logit_instance+math.log(39./252.), pseudo_label_positive))/torch.sum(mask)
                    cnt_over_thresh+=1
        
            loss = loss_bag + loss_instance
        else:
            loss = loss_bag

        return loss, cnt_positive, cnt_over_thresh
    
    model.train()
    for i, (images, target) in enumerate(train_loader):
        # images --> #bags x #instances x #dims
        images = images.type(torch.FloatTensor).to(args.device, non_blocking=True)
        # target --> #bags x #classes
        target = target.type(torch.FloatTensor).to(args.device, non_blocking=True)

        # First step
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss, cnt_positive_1, cnt_over_thresh_1 = train_single(model=model, images=images, target=target, epoch=epoch, cnt_positive=cnt_positive_1, cnt_over_thresh=cnt_over_thresh_1, criterion_mean=criterion_mean, criterion_none=criterion_none)
        
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        
    print(f'[1]: {cnt_over_thresh_1}/{cnt_positive_1}, [2]: {cnt_over_thresh_2}/{cnt_positive_2}')

def validate(val_loader, model, args):
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
                # output --> #bags x #classes
                logit_bag, _ = model(images)
            #classes  (prob)
            bag_predictions.append(torch.sigmoid(logit_bag.type(torch.DoubleTensor)).squeeze(0).cpu().numpy())

        # bag_labels --> #bag x #classes
        bag_labels = np.array(bag_labels)
        # bag_predictions --> #bag x #classes
        bag_predictions = np.array(bag_predictions)
        assert len(bag_predictions.shape) == 2
        auc, acc = multi_label_roc(bag_labels, bag_predictions, num_classes=bag_labels.shape[-1], pos_label=1)

    return auc, acc

if __name__ == '__main__':
    args = parser.parse_args()
    # txt_name = f'{args.dataset}_{args.pretrain_type}_downstreamLR_{args.lr}_optimizer_{args.optimizer}_epoch{args.epochs}_wd{args.weight_decay}'
    txt_name = f'Transformer_Transpose_{datetime.today().strftime("%m%d")}_{args.dataset}_{args.pretrain_type}_epoch{args.epochs}_wd{args.weight_decay}_scheduler_{args.scheduler}_'+\
    f'thresh_{args.pseudo_prob_threshold}_semiepoch_{args.semi_start_epoch}_share_{args.share_proj}_nl_{args.num_layers}'
    
    acc_fold_tr = []
    auc_fold_tr = []

    acc_fold_val = []
    auc_fold_val = []

    acc_fold_test = []
    auc_fold_test = []

    args.num_classes=2 if args.dataset=='tcga_lung' else 1
    args.device = 0

    t_start = time.time()
    for fold_num in range(1, args.fold+1):
        auc_test, acc_test, auc_val, acc_val, auc_tr, acc_tr, category_idx, epoch_best = run_fold(args, fold_num)
        
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
        f.write(f'==== LR-pretrain: {args.pretrain_type} || LR-down: {args.lr} || Optimizer: {args.optimizer} || scheduler: {args.scheduler} ||\n')
        
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