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
parser.add_argument('--data-root', default='/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50', help='path to dataset')
parser.add_argument('--fold', default=5, help='number of fold for cross validation')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 1)')
parser.add_argument('--scheduler', default='single', choices=['single', 'multi'], type=str, help='loss scheduler')
parser.add_argument('--loss', default='bce', choices=['bce'], type=str, help='loss function')
parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='the total batch size on the current node (DDP)')
parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

parser.add_argument('--dataset', default='CAMELYON16', choices=['CAMELYON16', 'tcga_lung', 'tcga_stad'], type=str, help='dataset type')
# parser.add_argument('--pretrain-type', default='ImageNet_Res50_im', help='weight folder')
# parser.add_argument('--pretrain-type', default='simclr_lr1', help='weight folder')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'], type=str, help='optimizer')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR', help='initial learning rate', dest='lr')
# DTFD: 1e-4, TransMIL: 1e-5
parser.add_argument('--weight-decay', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)', dest='weight_decay')
parser.add_argument('--mil-model', default='MilTransformer', choices=['MilTransformer', 'monai.max','monai.att','monai.att_trans','milmax', 'milmean', 'Attention', 'GatedAttention','Dsmil','milrnn','Dtfd'], type=str, help='use pre-training method')
parser.add_argument('--if-learn-instance', default=False, help='if_learn_instance')
parser.add_argument('--share-proj', default=False, help='if share projection')
parser.add_argument('--pseudo-prob-threshold', default=0.8, type=float, help='pseudo_prob_threshold')
parser.add_argument('--layerwise-shuffle', default=False, help='Shuffle')
parser.add_argument('--n-head', default=2, type=int, help='Number of head')
parser.add_argument('--sr-ratio', default=8, type=int, help='self-attention grouping ratio')
parser.add_argument('--sr-ratio', default=8, type=int, help='self-attention grouping ratio')
parser.add_argument('--if-balance-param', default=False, help='balance_param')

parser.add_argument('--pushtoken', default=False, help='Push Bullet token')

def run_fold(args, fold, txt_name) -> Tuple:

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # cudnn.deterministic = True

    dataset_train = Dataset_pkl(path_pretrained_pkl_root=args.data_root, fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='train', num_classes=args.num_classes, seed=args.seed)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    dataset_val = Dataset_pkl(path_pretrained_pkl_root=args.data_root, fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='val', num_classes=args.num_classes, seed=args.seed)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    args.num_step = len(loader_train)
    if 'Res18' in args.data_root:
        dim_in = 512
    else:
        dim_in = 2048

    if 'monai' in args.mil_model:
        mode = args.mil_model.split('.')[-1]
        model = milmodels.__dict__['MonaiMil'](dim_in=dim_in, dim_latent=512, dim_out=args.num_classes, mil_mode=mode).cuda()
    elif args.mil_model == 'MilTransformer':
        model = milmodels.__dict__[args.mil_model](args=args, if_learn_instance=args.if_learn_instance, pseudo_prob_threshold=args.pseudo_prob_threshold, share_proj=args.share_proj, optimizer=None, criterion=None, scheduler=None, dim_in=dim_in, dim_latent=512, dim_out=args.num_classes).cuda()
    else:
        model = milmodels.__dict__[args.mil_model](args=args, optimizer=None, criterion=None, scheduler=None, dim_in=dim_in, dim_latent=512, dim_out=args.num_classes).cuda()
    
    # if args.loss == 'bce':
    #     criterion = nn.BCEWithLogitsLoss().cuda()
        
    # if args.optimizer == 'adam':
    #     optimizer = torch.optim.Adam(model.parameters(), 0, weight_decay=args.weight_decay)
    # elif args.optimizer == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), 0, momentum=args.momentum, weight_decay=args.weight_decay)
    # elif args.optimizer == 'adamw':
    #     optimizer = torch.optim.AdamW(model.parameters(), 0, weight_decay=args.weight_decay)
    
    

    # # 고쳐야 하나..?
    # if args.scheduler == 'single':
    #     scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loader_train))
    # elif args.scheduler == 'multi':
    #     scheduler = CosineAnnealingWarmUpRestarts(optimizer, eta_max=args.lr, step_total=args.epochs * len(loader_train))

    auc_best = 0.0
    epoch_best = 0
    file_name = f'{txt_name}_lr{args.lr}_fold{fold}.pth'
    for epoch in trange(1, (args.epochs+1)):        
        train(loader_train, model)
        auc, acc = validate(loader_val, model, args)
        if np.mean(auc) > auc_best:
            epoch_best = epoch
            auc_best = np.mean(auc)
            auc_val = auc
            acc_val = acc
            torch.save({'state_dict': model.state_dict()}, file_name)
        print(f'auc val: {auc}')
        print(f'kk')
    

    dataset_test = Dataset_pkl(path_pretrained_pkl_root=args.data_root, fold_now=999, fold_all=9999, shuffle_slide=False, shuffle_patch=False, split='test', num_classes=args.num_classes, seed=args.seed)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    checkpoint = torch.load(file_name, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])
    os.remove(file_name)

    auc_test, acc_test = validate(loader_test, model, args)
    auc_tr, acc_tr = validate(loader_train, model, args)

    del dataset_train, loader_train, dataset_val, loader_val
    print(f'fold [{fold}]: epoch_best ==> {epoch_best}')
    print(f'auc_tr: {auc_tr}, auc_val: {auc_val}, auc_test: {auc_test},')
    
    return auc_test, acc_test, auc_val, acc_val, auc_tr, acc_tr, dataset_test.category_idx, epoch_best

def train(train_loader, model):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        # images --> #bags x #instances x #dims
        images = images.type(torch.FloatTensor).to(args.device, non_blocking=True)
        # target --> #bags x #classes
        target = target.type(torch.FloatTensor).to(args.device, non_blocking=True)

        model.update(images, target)

        # # First step
        # optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        #     loss = model.calculate_objective(images, target)

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # scheduler.step()

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
                prob_bag, _ = model.infer(images)
            #classes  (prob)
            bag_predictions.append(prob_bag.squeeze(0).cpu().numpy())

        # bag_labels --> #bag x #classes
        bag_labels = np.array(bag_labels)
        # bag_predictions --> #bag x #classes
        bag_predictions = np.array(bag_predictions)
        assert len(bag_predictions.shape) == 2
        auc, acc = multi_label_roc(bag_labels, bag_predictions, num_classes=bag_labels.shape[-1], pos_label=1)

    return auc, acc

if __name__ == '__main__':
    args = parser.parse_args()
    if args.if_balance_param:
        if args.dataset == 'CAMELYON16':
            args.balance_param = math.log(111./159.)
        elif args.dataset == 'tcga_lung':
            args.balance_param = math.log(371./380.)    
        elif args.dataset == 'tcga_stad':
            args.balance_param = math.log(39./252.)
    else:
        args.balance_param = 0.0
    args.pretrain_type = args.data_root.split("/")[-2:]
    # txt_name = f'{args.dataset}_{args.pretrain_type}_downstreamLR_{args.lr}_optimizer_{args.optimizer}_epoch{args.epochs}_wd{args.weight_decay}'    
    txt_name = f'{datetime.today().strftime("%m%d")}_{args.dataset}_{args.mil_model}_epoch{args.epochs}_share_proj{args.share_proj}_' +\
    f'if_learn_instance{args.if_learn_instance}_pseudo_prob_threshold{args.pseudo_prob_threshold}_n_head{args.n_head}_sr_ratio{args.sr_ratio}_layerwise_shuffle{args.layerwise_shuffle}'

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
        f.write(f'===================== LR-pretrain: {args.pretrain_type} || LR-down: {args.lr} =======================\n')
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