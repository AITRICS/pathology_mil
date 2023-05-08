import os
import time
import multiprocessing as mp
from itertools import repeat
import argparse
import glob
from openslide import OpenSlide
import cucim 
import cv2
import math
import numpy as np
import pandas as pd
import pickle
from PIL import Image


# def generate_patch(path_wsi, pkl_dict, args):
def generate_patch(path_wsi, pkl_dict, args):
    """
    0) Get mpp_np
    1) Get level_mask, level_patch, size_patch_level_patch
    2) Get threshold_cb, threshold_cr using level_mask
    3) Do for loop using size_patch_level_patch
    4) (Under for loop) Get if_background
    5) (Under for loop & Under if if_background) Save patch
    """
    # path_subjectID = os.path.join(args.config[args.dataset]['root_patch'], args.foldername, pkl_dict[os.path.basename(path_wsi).split('.')[0]])
    path_subjectID = os.path.join(args.config[args.dataset]['root_patch'], pkl_dict[os.path.basename(path_wsi).split('.')[0]])
    if not os.path.exists(path_subjectID):
        os.makedirs(path_subjectID)

    wsi = OpenSlide(path_wsi)
    ## 0) Get mpp_np
    # mppx and mppy are always almost-same
    if path_wsi[-3:] == 'tif':
        try:
            mppx = float(10**4) / float(wsi.properties['tiff.XResolution'])
            mppy = float(10**4) / float(wsi.properties['tiff.YResolution'])
        except:
            mppx = float(wsi.properties['openslide.mpp-x']) 
            mppy = float(wsi.properties['openslide.mpp-y'])                 
    else:
        try:
            mppx = float(wsi.properties['openslide.mpp-x'])
            mppy = float(wsi.properties['openslide.mpp-y'])
        except:                
            raise ValueError(f'something wrong !!!!!!!!!!:\n{path_wsi}\nhas no mpp info')
    
    mppx_np = mppx * np.array(list(wsi.level_downsamples))
    mppy_np = mppy * np.array(list(wsi.level_downsamples))

    ## 1) Get level_mask, level_patch, size_patch_level_patch
    level_mask = np.argmin(np.absolute(mppx_np-args.config[args.dataset]['mpp_mask']))
    assert level_mask == (np.argmin(np.absolute(mppy_np-args.config[args.dataset]['mpp_mask'])))

    if args.dataset == 'tcga_lung':
        if (mppx >= 0.25) or (mppy >= 0.25):
            level_patch = 0
        else:
            level_patch = 1
    else:
        level_patch = np.argmin( 1.0/(mppx_np-args.config[args.dataset]['mpp_patch']) )
        assert level_patch == (np.argmin( 1.0/(mppy_np-args.config[args.dataset]['mpp_patch']) ))

        # mpp of level_patch must be smaller (more zoomed-in) than mpp_patch (desired mpp for patch, eg, 0.5 for camelyon16)
        assert args.config[args.dataset]['mpp_patch'] > mppx_np[level_patch]
        assert args.config[args.dataset]['mpp_patch'] > mppy_np[level_patch]
    

    size_patch_level_patchx = round(args.config[args.dataset]['size_patch'] * args.config[args.dataset]['mpp_patch'] / mppx_np[level_patch])
    size_patch_level_patchy = round(args.config[args.dataset]['size_patch'] * args.config[args.dataset]['mpp_patch'] / mppy_np[level_patch])

    ## 2) Get threshold_cb (th_cb, 크면 foreground), threshold_cr (th_cr, 크면 foreground) using level_mask

    mask_pl = wsi.read_region(location=(0,0), level=level_mask,
                        size=(int(wsi.properties[f'openslide.level[{level_mask}].width']), int(wsi.properties[f'openslide.level[{level_mask}].height']))).convert('YCbCr')
    mask_np = np.array(mask_pl)
    th_cb, _ = cv2.threshold(mask_np[:, :, 1], 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th_cr, _ = cv2.threshold(mask_np[:, :, 2], 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ## 3) Get list_idx_width, list_idx_height, threshold_background. 0 dimension of numpy is height

    w_level_patch, h_level_patch = wsi.level_dimensions[level_patch]
    list_idx_width = list(range(0, w_level_patch, size_patch_level_patchx))
    list_idx_height = list(range(0, h_level_patch, size_patch_level_patchy))

    if w_level_patch % size_patch_level_patchx != 0:
        list_idx_width.pop(-1)
        list_idx_width.append(w_level_patch-size_patch_level_patchx)
    if h_level_patch % size_patch_level_patchy != 0:
        list_idx_height.pop(-1)
        list_idx_height.append(h_level_patch-size_patch_level_patchy)

    threshold_background = size_patch_level_patchx * size_patch_level_patchy * args.threshold_mask * 255
    kernel = np.ones((32, 32), np.uint8)

    ## 4) Do for loop using size_patch_level_patch
    for i in list_idx_width:
        for j in list_idx_height:
            patch_pl = wsi.read_region(location=(i,j), level=level_patch, size=(size_patch_level_patchx, size_patch_level_patchy))
            patch_ycbcr_np = np.array(patch_pl.convert('YCbCr'))
    ## 5) Get if_background
            
            mask_cb = cv2.threshold(patch_ycbcr_np[:,:,1], thresh=th_cb, maxval=1, type=cv2.THRESH_BINARY)
            mask_cr = cv2.threshold(patch_ycbcr_np[:,:,2], thresh=th_cr, maxval=1, type=cv2.THRESH_BINARY)
            mask = mask_cb[1] | mask_cr[1]
            
            mask = cv2.erode(cv2.dilate(mask, kernel, iterations=1), kernel, iterations=1)
            mask = cv2.dilate(cv2.erode(mask, kernel, iterations=1), kernel, iterations=1)
            print(f'min({np.min(mask)}), max({np.max(mask)})')
    # 4) (Under for loop) Get if_background
            if mask.sum() >= threshold_background:

    # 5) (Under for loop & Under if if_background) Save patch    
                # patch_pl.resize((args.config[args.dataset]['size_patch'], args.config[args.dataset]['size_patch']), Image.LANCZOS).save(os.path.join(args.config[args.dataset]['root_patch'], args.foldername, pkl_dict[os.path.basename(path_wsi).split('.')[0]], f'w{i}_h{j}.jpeg'))
                patch_pl = patch_pl.resize((args.config[args.dataset]['size_patch'], args.config[args.dataset]['size_patch']), Image.LANCZOS).convert('RGB')
                patch_pl.save(os.path.join(args.config[args.dataset]['root_patch'], pkl_dict[os.path.basename(path_wsi).split('.')[0]], f'w{i}_h{j}.jpeg'))

def main(args):

    path_folder = args.config[args.dataset]['root_patch']
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)

    with open(f'preprocess/{args.config[args.dataset]["pkl"]}_train.pkl', 'rb') as fp:
        pkl_tr = pickle.load(fp)
    with open(f'preprocess/{args.config[args.dataset]["pkl"]}_test.pkl', 'rb') as fp:
        pkl_test = pickle.load(fp)

    pkl = {}
    pkl.update(pkl_tr)
    pkl.update(pkl_test)

    pathlist_wsi = glob.glob(args.config[args.dataset]['path_list_wsi'])

    # check if any subject is missing
    for _subjectID in pkl.keys():
        if_wsi_exist = False
        for path_wsi in pathlist_wsi:
            if os.path.basename(path_wsi).split('.')[0] == _subjectID:
                if_wsi_exist = True
        if if_wsi_exist == False:
            raise ValueError(f'WSI file does NOT exist for the followin subject ID: {_subjectID}')
    
    for path_wsi in pathlist_wsi:
        if_pkl_exist = False
        _subjectID = os.path.basename(path_wsi).split('.')[0]
        for _subjectID_pkl in pkl.keys():
            if _subjectID == _subjectID_pkl:
                if_pkl_exist = True
        if if_pkl_exist == False:
            pathlist_wsi.remove(path_wsi)

    p = mp.Pool(os.cpu_count())
    p.starmap(generate_patch, zip(pathlist_wsi, repeat(pkl), repeat(args)))
    # for path_wsi in pathlist_wsi:
    #     generate_patch(path_wsi, pkl, args)


def parse_args():

    parser = argparse.ArgumentParser(description="Generate Patches from WSI")

    # parser.add_argument("--root-wsi", default = None, help = 'path to root folder of WSI image')
    parser.add_argument("--foldername", default="tcga_stad_ex", help="foldername")
    parser.add_argument("--dataset", default='tcga_stad', choices=['camelyon', 'tcga_lung', 'tcga_stad'], help="dataset type")
    # parser.add_argument("--patch_size", default = 256, type =int, help="The number of pixel that you want for the patches")
    # parser.add_argument("--level-mask", default = 3, type= int)
    # parser.add_argument("--level-patch", default = 1, type= int)
    parser.add_argument("--threshold-mask", default = 0.25, type= float, help = 'minimum portion of foreground to keep certain patch')
    # parser.add_argument("--mpp", default= 1, type = float, help= 'desired mpp')
    # parser.add_argument("--multiprocess", action="store_true", help = 'Use multiprocessing while extracting patch from WSI')
    # parser.add_argument("--num-process", default=20, type=int)
    args = parser.parse_args()
    return args

# min(level count):
# Camelyon16: 8, 28.97 mpp --> 30 mpp와 가장 가까운 mpp 레벨에서 마스크 만들기
# tcga_lung: 2, 1.966 mpp --> 2 mpp와 가장 가까운 mpp 레벨에서 마스크 만들기
# tcga_stad: 2, 1.975 mpp --> 2 mpp와 가장 가까운 mpp 레벨에서 마스크 만들기.
#  그런데 그냥 위쪽 레벨로 (마스크 레벨) threshold만 구하고 마스크는 안 구하면 안되나? threshold 있으면 패치상에서도 할 수 있으니.
# 레벨당 마스크 threshold 바뀌는지 확인 -> openslide 내에서는 별로 안바뀜 (마지막에서 두번째 레벨 쓰자)

if __name__ == "__main__":
    run_start = time.time()
    args = parse_args()
    args.config = {'camelyon': {'mpp_mask': 30, 'mpp_patch': 0.5, 'size_patch': 256, 'ext': '.tif', 'path_list_wsi': '/nfs/thena/shared/camelyon/CAMELYON16/images/*', 'root_patch': '/nfs/thena/shared/', 'pkl': 'subjectID_camelyon16'},
                'tcga_lung': {'mpp_mask': 2, 'mpp_patch': 0.5, 'size_patch': 256, 'ext': '.svs', 'path_list_wsi': '/nfs/thena/shared/tcga_lung/*/*', 'root_patch': '/nfs/thena/shared/', 'pkl': 'subjectID_tcgalung'},
                'tcga_stad': {'mpp_mask': 2, 'mpp_patch': 1.0, 'size_patch': 512, 'ext': '.svs', 'path_list_wsi': '/nfs/thena/shared/tcga_stad_wsi/*/*', 'root_patch': '/nfs/thena/shared/', 'pkl': 'subjectID_tcgastadher2'}}
        
    args.config[args.dataset]['root_patch'] += args.foldername

    if os.path.isdir(args.config[args.dataset]['root_patch']):
        raise ValueError('Invalid foldername!!! Foldername already exists!!')

    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")
    main(args)
    print(f'All done in total {time.time() - run_start} sec')

# https://github.com/YuZhang-SMU/Cancer-Prognosis-Analysis/blob/main/DC_MIL%20Code/preprocessing/extract_tiles.py
# https://github.com/binli123/dsmil-wsi/blob/master/deepzoom_tiler.py

# /mnt/aitrics_ext/ext01/shared/camelyon/CAMELYON16/images/{'normal_xxx.tif', 'tumor_xxx.tif', 'test_xxx.tif'}
# /mnt/aitrics_ext/ext01/shared/tcga_lung/{'AD', 'SC'}/{'TCGA-49-4512-01Z-00-DX7.svs'}
# /mnt/aitrics_ext/ext01/shared/tcga_stad_wsi/{'amplified', 'normal'}/{'TCGA-D7-6820-01Z-00-DX2.svs'}


# data = {
#     'subjectID_camelyon16_train': '/nfs/thena/shared/cam_jpeg/train/*/*',
#     'subjectID_camelyon16_test': '/nfs/thena/shared/cam_jpeg/test/*/*',
    
#     'subjectID_tcgalung_train': '/nfs/thena/shared/tcga_lung_jpeg/train/fold*/*/*',
#     'subjectID_tcgalung_test': '/nfs/thena/shared/tcga_lung_jpeg/test/*/*',
    
#     'subjectID_tcgastadher2_train': '/nfs/thena/shared/tcga_stad_jpeg/train/fold*/*/*',
#     'subjectID_tcgastadher2_test': '/nfs/thena/shared/tcga_stad_jpeg/test/*/*'
#     }

# /nfs/thena/shared/camelyon/CAMELYON16/images/*