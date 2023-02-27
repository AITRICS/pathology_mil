import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from utils import rnndata, adjust_learning_rate, loss, Dataset_pkl, CosineAnnealingWarmUpSingle, CosineAnnealingWarmUpRestarts, optimal_thresh, multi_label_roc, save_checkpoint
from models import MilBase, rnn_single
import numpy as np
from tqdm import trange

parser = argparse.ArgumentParser(description='MIL Training')
parser.add_argument('--data-root', default='/nfs/thena/shared/pathology_mil', help='path to dataset')
parser.add_argument('--fold', default=5, help='number of fold for cross validation')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 1)')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
parser.add_argument('--dataset', default='tcga_stad', choices=['CAMELYON16', 'tcga_lung', 'tcga_stad'], type=str, help='dataset type')
parser.add_argument('--pretrain-type', default='simclr_lr1', help='weight folder')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='the total batch size on the current node (DDP)')
parser.add_argument('--model_path', default=None, type=str)
# rnn
parser.add_argument('--r-bs', default=128, type=int)
parser.add_argument('--r-epoch', default=100, type=int)

# parser.add_argument('--scheduler', default='single', choices=['single', 'multi'], type=str, help='loss scheduler')
# parser.add_argument('--loss', default='bce', choices=['bce'], type=str, help='loss function')
# parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
# parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--optimizer', default='adamw', choices=['sgd', 'adam', 'adamw'], type=str, help='optimizer')
# parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
# DTFD: 1e-4, TransMIL: 1e-5
#parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
#parser.add_argument('--mil-model', default='MilBase', choices=['MilBase'], type=str, help='use pre-training method')
#parser.add_argument('--agg', default='max', choices=['max', 'mean', 'rnn'])

def run_fold(args, fold):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    
    # our dataset & dataloader
    dataset_train = Dataset_pkl(path_fold_pkl=os.path.join(args.data_root, 'cv', args.dataset), path_pretrained_pkl_root=os.path.join(args.data_root, 'features', args.dataset, args.pretrain_type), fold_now=fold, fold_all=args.fold, shuffle_slide=True, shuffle_patch=True, split='train', num_classes=args.num_classes, seed=args.seed)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dataset_val = Dataset_pkl(path_fold_pkl=os.path.join(args.data_root, 'cv', args.dataset), path_pretrained_pkl_root=os.path.join(args.data_root, 'features', args.dataset, args.pretrain_type), fold_now=fold, fold_all=args.fold, shuffle_slide=False, shuffle_patch=False, split='val', num_classes=args.num_classes, seed=args.seed)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # pretrained encoder & fc
    model = MilBase(dim_out=args.num_classes, pool=nn.AdaptiveMaxPool1d((1))).cuda() 
    model.eval()
    state_dict = torch.load(args.model_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict['state_dict'], strict=False)
    assert missing_keys == [] and unexpected_keys == []
    
    # rnn model & optimizer
    rnn = rnn_single(128).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    
    # inference & collect rnn data
    train_bags = inference(loader_train, model)
    val_bags = inference(loader_val, model)
    rnn_train_dataset = rnndata(train_bags)
    rnn_val_dataset = rnndata(val_bags)
    rnn_train_loader = torch.utils.data.DataLoader(rnn_train_dataset, batch_size=args.r_bs, shuffle=True, num_workers=4, pin_memory=False)
    rnn_val_loader = torch.utils.data.DataLoader(rnn_val_dataset, batch_size=args.r_bs, shuffle=False, num_workers=4, pin_memory=False)

    # rnn train & validate
    for epoch in trange(args.r_epoch):
        train_loss = train_single(rnn, rnn_train_loader, criterion, optimizer)
    print(f'final train loss : {train_loss}')
    auc, acc = test_single(rnn, rnn_val_loader)
    print('validation acc : %.4f , pos auc : %.4f, neg auc : %.4f' %(acc, auc[1], auc[0]))
    
    return auc, acc

def inference(loader,model):
    bags=[] 
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = torch.Tensor(images).type(torch.FloatTensor).cuda()
            midfeats = model.encoder(images) # bs x patch x fs
            scores = model.score(midfeats).squeeze(-1) # bs x patch
            top_indices = torch.argsort(scores, dim=-1)[:,:10] # bs x s
            top_midfeats = torch.index_select(midfeats, dim=1, index=top_indices[0]).cpu() # assume bs ==1
            assert top_midfeats.shape == (1,10,512)
            if target == 1:
                t = torch.tensor([0.,1.])
            elif target == 0:
                t = torch.tensor([1.,0.])
            bags.append({'feats':top_midfeats, 'target' : t}) # bs x s x fs
    return bags        

def train_single(rnn, rnn_loader, criterion, optimizer):
    rnn.train()
    running_loss = 0.
    for j, (inputs, target) in enumerate(rnn_loader):
        rnn_bs = inputs.size(0)
        rnn.zero_grad()
        state = rnn.init_hidden(rnn_bs).cuda()
        for s in range(inputs[0].size(1)):
            input = inputs[:,:,s,:].squeeze().cuda() # bs x fs
            output, state = rnn(input, state) #output : bs x num_cls, state : bs x ndims
        target = target.cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*target.size(0)
    running_loss = running_loss/len(rnn_loader.dataset)  
    return running_loss

def test_single(rnn, rnn_loader):
    rnn.eval()
    bag_predictions = []
    bag_labels =[]
    with torch.no_grad():
        for j, (inputs, target) in enumerate(rnn_loader):
            rnn.zero_grad()
            rnn_bs = inputs.size(0) 
            state = rnn.init_hidden(rnn_bs).cuda()
            for s in range(inputs[0].size(1)): 
                input = inputs[:,:,s,:].squeeze().cuda() # bs x fs
                output, state = rnn(input, state) #output : bs x num_cls, state : bs x ndims
            bag_predictions += list(torch.sigmoid(output).cpu().numpy())
        target = target.numpy()
        auc, acc = multi_label_roc(target, np.array(bag_predictions), num_classes=target.shape[-1], pos_label=1)
    return auc, acc

if __name__ == '__main__':
    args = parser.parse_args()
    
    # debug 
    # args.model_path ='dataset_CAMELYON16_pretrain_simclr_lr30_lr_1e-08_fold_1.pth'
    # args.dataset = 'CAMELYON16'
    # args.pretrain_type = 'simclr_lr30'


    acc_fold = []
    auc_fold = []

    args.num_classes=2 if args.dataset=='tcga_lung' else 1
    args.device = 0

    t_start = time.time()
    for fold_num in range(1, args.fold+1):
        _auc, _acc = run_fold(args, fold_num)
        acc_fold.append(_acc)
        auc_fold.append(_auc)
    print(f'Training took {round(time.time() - t_start, 3)} seconds')