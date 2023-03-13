import os
import time
import multiprocessing as mp
from itertools import repeat
import argparse
import glob
import openslide
import cucim 
import cv2
import math
import numpy as np
import pandas as pd

def get_mask(args,item):
    wsi = openslide.OpenSlide(item)
    cu_wsi = cucim.clara.CuImage(item)
        
    if args.mask_level == None :
        mask_level = wsi.level_count -1
    else :
        mask_level = args.mask_level
    maskimg = np.array(cu_wsi.read_region(location = (0,0),level = mask_level, num_workers= args.num_workers )) 
    
    # otsu thresholding in ycrcb domain
    cvimg = cv2.cvtColor(maskimg[:,:,:3],cv2.COLOR_RGB2YCrCb)
    _, th1 = cv2.threshold(cvimg[:, :, 1], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(cvimg[:, :, 2], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = th1 | th2                     
    mask = cv2.resize(mask, dsize=wsi.level_dimensions[args.patch_level], interpolation=cv2.INTER_CUBIC)       
              
    return wsi, mask, mask_level, cu_wsi

def mapping(args, wsi, mask, mask_level, cu_wsi):
    try:
        mpp_ = wsi.properties['openslide.mpp-x']
    except :
        print(f'{args.imgName} : no mpp info')
        return
    
    if args.data == '16':
        ref = pd.read_csv('reference.csv', names=['name', 'class', 'subclass', 'binary'])
        if 'test' in args.imgName :
            _class = ref['class'][ref['name'].str.contains(args.imgName)]
    if args.data == 'tcga':
        if not os.path.isdir(os.path.join(args.dest, args.imgName)):
            os.makedirs(os.path.join(args.dest, args.imgName), exist_ok=True)

    hp_0 = args.patch_size * args.mpp / float(mpp_) 
    hp = int(hp_0 / wsi.level_downsamples[args.patch_level]) # patch size for high resolution
    mp = int(hp_0 / wsi.level_downsamples[mask_level]) # patch size for low resolution otsu mask
    if args.large_patch_size != None:
        hp_0_large = args.large_patch_size * args.mpp / float(mpp_)
        hp_large = int(hp_0_large / wsi.level_downsamples[args.patch_level])

    count = 0
    w,h = wsi.level_dimensions[mask_level] 
    for i in range(math.floor(w/mp)):
        for j in range(math.floor(h/mp)):
            if mask[j*hp:(j+1)*hp, i*hp:(i+1)*hp].sum() >= hp*hp* args.mask_threshold*255 :
                if args.large_patch_size is None:
                    patch = np.array(cu_wsi.read_region(location = (hp_0*i, hp_0*j),level = args.patch_level, size = (hp,hp), num_workers = 5))[:,:,:3]
                    resized = cv2.resize(patch, dsize=(args.patch_size, args.patch_size))
                else :
                    patchlev_lefttop = (i*hp + (hp-hp_large)/2, j*hp + (hp-hp_large)/2)
                    lev0_lefttop = (patchlev_lefttop[0]* wsi.level_downsamples[args.patch_level], patchlev_lefttop[1]* wsi.level_downsamples[args.patch_level])
                    patch = np.array(cu_wsi.read_region(location = lev0_lefttop ,level = args.patch_level,size = (hp_large,hp_large), num_workers = 5))[:,:,:3]
                    resized = cv2.resize(patch, dsize=(args.large_patch_size, args.large_patch_size))                
                count +=1
                
                if args.data == 'tcga':
                    cv2.imwrite(os.path.join(args.dest, args.imgName,'patch_{}.jpeg'.format(count)),resized)
                else : # camelyon16
                    if 'test' in args.imgName :
                        _class = ref['class'][ref['name'].str.contains(args.imgName)]
                        if not os.path.isdir(os.path.join(args.dest,"test",_class.item(), args.imgName)):
                            os.makedirs(os.path.join(args.dest,"test",_class.item(), args.imgName))
                        cv2.imwrite(os.path.join(args.dest,"test",_class.item(), args.imgName,'patch_{}.jpeg'.format(count)),resized)

                    elif 'normal' in args.imgName : 
                        if not os.path.isdir(os.path.join(args.dest,"train/normal", args.imgName)) :
                            os.makedirs(os.path.join(args.dest,"train/normal", args.imgName))
                        cv2.imwrite(os.path.join(args.dest,"train/normal", args.imgName,'patch_{}.jpeg'.format(count)),resized)
                    elif 'tumor' in args.imgName : 
                        if not os.path.isdir(os.path.join(args.dest,"train/tumor", args.imgName)) :
                            os.makedirs(os.path.join(args.dest,"train/tumor", args.imgName))
                        cv2.imwrite(os.path.join(args.dest,"train/tumor", args.imgName,'patch_{}.jpeg'.format(count)),resized)         
                                            
    print(f'Patch collection is done for {args.imgName}, {count} patches are collected')
        
def iter_data(item, args):
    if args.data == 'tcga':
        args.imgName = item.split('/')[-2] + '/' + item.split('/')[-1].split('.')[0] # including labels
    else : # camelyon16
        args.imgName = item.split('/')[-1].split('.')[0]
    
    print("*************{}**************".format(args.imgName))
    wsi, mask, mask_level, cu_wsi = get_mask(args, item)
    mapping(args,wsi, mask, mask_level, cu_wsi)
        


def patch_generation(args): 
    if args.data == 'tcga':
        data_list = glob.glob(args.wsi_root+'/*/*') 
    else : # camelyon16
        data_list = glob.glob(args.wsi_root+'/*')
    print(f'{len(data_list)} slides found')
    
    if args.multiprocess :
        p = mp.Pool(args.num_process)
        p.starmap(iter_data,zip(data_list,repeat(args)))
    else : 
        for item in data_list :
            iter_data(item, args)

def parse_args():

    parser = argparse.ArgumentParser(description="Multiple Instance Learning (MIL) example of classification from WSI.")
 
    parser.add_argument("--imgName", default=None , type= str)
    parser.add_argument("--wsi_root", default = None, help = 'path to root folder of CAMELYON WSI image')
    parser.add_argument("--data", choices=['CAMELYON16','tcga'], type = str)
    parser.add_argument("--dest", default="/mnt/aitrics_ext/ext01/shared/tcga_stad_wsi", help="path to root folder of camelyon images")

    parser.add_argument("--patch_size", default = 256, type =int, help="The number of pixel that you want for the patches")
    parser.add_argument("--large_patch_size", default= None, type = int)
    parser.add_argument("--mask_ratio", default = None, type= float, help='ratio between level 0 and mask level')
    parser.add_argument("--patch_ratio", default = None, type= float, help = 'ratio between level 0 and patch level')
    parser.add_argument("--mask_level", default = None, type= int)
    parser.add_argument("--patch_level", default = 1, type= int)
    parser.add_argument("--mask_threshold", default = 0.75, type= float, help = 'minimum portion of foreground to keep certain patch')
    parser.add_argument("--mpp", default= 1, type = float, help= 'desired mpp')

    parser.add_argument("--multiprocess", action="store_true", help = 'Use multiprocessing while extracting patch from WSI')
    parser.add_argument("--num_process", default=20, type=int)
    parser.add_argument("--passifdone", action= "store_true", help = "Skip preprocessing if already done")
    parser.add_argument("--num_workers",default=4, type=int, help='the number of workers for reading whole slide image')
    args = parser.parse_args()
    return args
    
if __name__ =="__main__":
    '''
    wsi-root
    camelyon16 : 하위에 바로 wsi
    tcga : 하위에 label/wsi
    '''
    run_start = time.time()
    args = parse_args()
    
    #debug
    args.data = 'CAMELYON16'
    args.patch_level = 1
    args.mpp = 1
    args.patch_size = 256
    args.large_patch_size = 384
    args.dest = '.'
    args.wsi_root = '/nfs/thena/shared/camelyon/CAMELYON16/images'
    # args.multiprocess = True
    
    
    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")
    patch_generation(args)
    print(f'All done in total {time.time() - run_start} sec')

