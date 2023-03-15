import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as torchmp
from monai.data import Dataset
from monai.data.image_reader import NumpyReader, PILReader
#from monai.networks.nets import milmodel
from monai.transforms import Compose, RandFlip, RandRotate90, ScaleIntensityRange, ToTensor, LoadImage
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from train import train_epoch, val_epoch
from utils import save_checkpoint, c_datalist, load_model
from torchvision.transforms import CenterCrop, PILToTensor, RandomRotation, RandomCrop
import builder.resnet as resnets
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import datetime


def main_worker(gpu, args):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cudnn.deterministic = True
    
    ckptname = str(args.checkpoint).split('/')[-1].split('.')[0] if args.checkpoint !=0 else ''
    date = datetime.date.today().strftime("%m%d%Y")
    args.gpu = gpu
    args.file_name = f'{date}_exp_milmode_{args.mil_mode}_epochs{args.epochs}_batchsize{args.batch_size}_lr{args.optim_lr}_weightdecay{args.weight_decay}_ckpt_{ckptname}'

    if args.distributed:
        args.rank = args.rank * torch.cuda.device_count() + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,  world_size=args.world_size, rank=args.rank)
        
    print(args.rank, " gpu", args.gpu)
    torch.cuda.set_device(args.gpu)  # use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.epochs)
    
    training_list, validation_list, test_list = c_datalist(args)
    if args.test :
        validation_list = test_list 
        args.validate = True
        print('TEST MODE !')
        
    transform = Compose(
        [
            RandomRotation(180),
            CenterCrop(256), # c, h, w
            RandomCrop(224),
            PILToTensor(), # h, w, c
            ScaleIntensityRange(a_min=np.float32(0), a_max=np.float32(255)),
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate90(prob=0.5, max_k=3),
            ToTensor()
        ],
        map_items=True
    )
    
    transform_valid = Compose(
        [
            CenterCrop(224), # c, h, w
            PILToTensor(), # h, w, c
            ScaleIntensityRange(a_min=np.float32(0), a_max=np.float32(255)),
        ],
        map_items=True
    )
        
    if args.quick:  # for debugging on a small subset
        training_list = training_list[0:7]
        validation_list = validation_list[0:5] #라벨 섞여야 auroc 나옴
        args.epochs = 2

    train_sampler = DistributedSampler(training_list) if args.distributed else None
    val_sampler = DistributedSampler(validation_list, shuffle=False) if args.distributed else None

    train_loader = torch.utils.data.DataLoader(training_list,batch_size=args.batch_size,shuffle=(train_sampler is None),num_workers=0,pin_memory=True,sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(validation_list,batch_size=1,shuffle=False,num_workers=0,sampler=val_sampler)

    if args.rank == 0:
        print("Dataset training:", len(training_list), "validation:", len(validation_list))


    model = load_model(args)

    best_acc = 0
    start_epoch = 0
        
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    if args.validate:
        epoch_time = time.time()
        val_loss, val_acc, val_f1, val_auroc, val_auprc, tn, fp, fn, tp = val_epoch(model, transform_valid, valid_loader, epoch=0, args=args)
        if args.rank == 0:
            if args.test == False :
                print("***************Final validation","loss: {:.4f}".format(val_loss),"acc: {:.4f}".format(val_acc),"f1 score: {:.4f}".format(val_f1),"AUROC: {:.4f}".format(val_auroc),"AUPRC: {:.4f}".format(val_auprc),"time {:.2f}s".format(time.time() - epoch_time))
            elif args.test == True : 
                print("***************Final Test","loss: {:.4f}".format(val_loss),"acc: {:.4f}".format(val_acc),"f1 score: {:.4f}".format(val_f1),"AUROC: {:.4f}".format(val_auroc),"AUPRC: {:.4f}".format(val_auprc),"time {:.2f}s".format(time.time() - epoch_time))
                print(f"    True negative : {tn}/{tn+fp+fn+tp}", f"    False positive : {fp}/{tn+fp+fn+tp}", f"    False negative : {fn}/{tn+fp+fn+tp}", f"    True positive : {tp}/{tn+fp+fn+tp}")    
        exit(0)
    params = model.parameters()

    if args.mil_mode in ["att_trans", "att_trans_pyramid"]:
        m = model if not args.distributed else model.module
        params = [
            {"params": list(m.attention.parameters()) + list(m.myfc.parameters()) + list(m.net.parameters())},
            {"params": list(m.transformer.parameters()), "lr": 6e-6, "weight_decay": 0.1},
        ]

    optimizer = torch.optim.AdamW(params, lr=args.optim_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)


    ###RUN TRAINING
    n_epochs = args.epochs
    val_acc_max = 0.0

    scaler = GradScaler(enabled=args.amp)

    for epoch in range(start_epoch, n_epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
            torch.distributed.barrier()

        print(args.rank, time.ctime(), "Epoch:", epoch)

        epoch_time = time.time()
        train_loss, train_acc, train_f1, train_auroc, train_auprc = train_epoch(model, transform, train_loader, optimizer, scaler=scaler, epoch=epoch, args=args)

        if args.rank == 0:
            print("Final training  {}/{}".format(epoch, n_epochs - 1),"loss: {:.4f}".format(train_loss),"acc: {:.4f}".format(train_acc),"f1 score: {:.4f}".format(train_f1),"AUROC: {:.4f}".format(train_auroc),"AUPRC: {:.4f}".format(train_auprc),"time {:.2f}s".format(time.time() - epoch_time))
        if args.distributed:
            torch.distributed.barrier()
 
        b_new_best = False
        val_acc = 0
        if (epoch >= 30 and epoch %5 ==0) or epoch ==args.epochs -1:

            epoch_time = time.time()
            val_loss, val_acc, val_f1, val_auroc, val_auprc,tn, fp, fn, tp = val_epoch(model, transform_valid, valid_loader, epoch=epoch, args=args)
            if args.rank == 0:
                print("***************Final validation  {}/{}".format(epoch, n_epochs - 1),"loss: {:.4f}".format(val_loss),"acc: {:.4f}".format(val_acc),"f1 score: {:.4f}".format(val_f1),"AUROC: {:.4f}".format(val_auroc),"AUPRC: {:.4f}".format(val_auprc),"time {:.2f}s".format(time.time() - epoch_time),)
                if val_acc > val_acc_max:
                    print("val acc ({:.6f} --> {:.6f})".format(val_acc_max, val_acc))
                    val_acc_max = val_acc
                    b_new_best = True
                    
        if args.rank == 0 and args.logdir is not None:
            save_checkpoint(model, epoch, args, best_acc=val_acc, filename=f'{args.file_name}.pt')
            if b_new_best:
                print(f"Copying best model. Epoch : {epoch}")
                shutil.copyfile(os.path.join(args.logdir,f'{args.file_name}.pt'), os.path.join(args.logdir, f'{args.file_name}_Best.pt'))
        scheduler.step()

    print("ALL DONE")


def parse_args():

    parser = argparse.ArgumentParser(description="Multiple Instance Learning (MIL) example of classification from WSI.")
 
    # data
    parser.add_argument("--imgName", default=None , type= str)
    parser.add_argument("--data", default = 'thyroid', type = str, help = 'Choose dataset among 16, 17, and thyroid')
    parser.add_argument("--c_patch_dir", default="/mnt/aitrics_ext/ext01/shared/hazel/cam_jpeg_train", help="path to root folder of camelyon images")
    parser.add_argument("--max_patch", default = 100, type = int, help = "maximum instances per one bag")
    parser.add_argument("--max_bag", default = 50, type = int, help = "maximum bag number per a single WSI")
    parser.add_argument("--margin", default = 1, type = int, help = "maximum possible difference between top and bottom embeddings")
    parser.add_argument("--_lambda", default = 1, type= int, help = "hyper parameter for loss trade-off")
    parser.add_argument("--patch_size", default = 256, type =int, help="The number of pixel that you want for the patches")
    
    # basic
    parser.add_argument("--num_classes", default=1, type=int, help="number of output classes")
    parser.add_argument("--epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size, the number of WSI images per gpu")
    parser.add_argument("--optim_lr", default=3e-5, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=0, type=float, help="optimizer weight decay")
    parser.add_argument("--amp", action="store_true", help="use AMP, recommended")
    parser.add_argument("--workers", default=4, type=int, help="number of workers for data loading")
    parser.add_argument("--checkpoint", default=None, help="load existing checkpoint")
    parser.add_argument("--checkpoint_from", default=None, choices=["MIL", "SIMCLR", "SIM_MIL"])
    parser.add_argument("--logdir", default='/nfs/thena/shared/pathology_mil/ckpts/CAMELYON16/', help="path to log directory to store Tensorboard logs")
    parser.add_argument("--file_name", default='exp')
    parser.add_argument("--pretrained", action ="store_true")
    
    # mode
    parser.add_argument("--mil_mode", default="att_trans", help="MIL algorithm")
    parser.add_argument("--validate",action="store_true",help="run only inference on the validation set, must specify the checkpoint argument")
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--quick", action="store_true", help="use a small subset of data for debugging")
    
    # multi gpu
    parser.add_argument("--distributed", action="store_true", help="use multigpu training, recommended")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://localhost:23456", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

    args = parser.parse_args()
    return args


'''
Possible Command for Training

python main.py --data 16 --mil_mode att_trans --distributed --amp --epoch 50 --optim_lr ????? --checkpoint ????? --checkpoint_from
python main.py --data 16 --mil_mode att_trans --distributed --amp --epoch 50 --optim_lr ????? --checkpoint ????? --validate

CUDA_VISIBLE_DEVICES=4 python main.py --data 16 --amp --epochs 50 --mil_mode 'att_trans' --optim_lr 0.01
'''

if __name__ == "__main__":
###################
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TORCH_CPP_LOG_LEVEL"] ="INFO"
###################
    random.seed(0)
    args = parse_args()
    
################## Debug
    # args.amp 
    # args.mil_mode = 'att_trans'
    # args.data ='16'
    #args.checkpoint = '/nfs/thena/shared/pathology_mil/ckpts/CAMELYON16/SIM_MIL/exp_milmode_att_trans_epochs50_batchsize1_lr0.001_weightdecay1e-05.pt'
    #args.checkpoint_from = 'SIM_MIL'
    # args.distributed = True
    # args.validate = True
#####################    
    
    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node * args.world_size

        print("Multigpu", ngpus_per_node, "rescaled lr", args.optim_lr)
        torchmp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
        torchmp.set_sharing_strategy('file_system')
        
    else:
        main_worker(0, args)