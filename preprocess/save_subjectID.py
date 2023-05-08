import glob
import pickle
import os
import random

# ROOT = {
#     'subjectID_camelyon16_train': '/mnt/aitrics_ext/ext01/shared/cam_jpeg/train/*/*',
#     'subjectID_camelyon16_test': '/mnt/aitrics_ext/ext01/shared/cam_jpeg/test/*/*',
    
#     'subjectID_tcgalung_train': '/mnt/aitrics_ext/ext01/shared/tcga_lung_jpeg/train/fold*/*/*',
#     'subjectID_tcgalung_test': '/mnt/aitrics_ext/ext01/shared/tcga_lung_jpeg/test/*/*',
    
#     'subjectID_tcgastadher2_train': '/mnt/aitrics_ext/ext01/shared/tcga_stad_jpeg/train/fold*/*/*',
#     'subjectID_tcgastadher2_test': '/mnt/aitrics_ext/ext01/shared/tcga_stad_jpeg/test/*/*'
#     }
ROOT = {
    'subjectID_camelyon16_train': '/nfs/thena/shared/cam_jpeg/train/*/*',
    'subjectID_camelyon16_test': '/nfs/thena/shared/cam_jpeg/test/*/*',
    
    'subjectID_tcgalung_train': '/nfs/thena/shared/tcga_lung_jpeg/train/fold*/*/*',
    'subjectID_tcgalung_test': '/nfs/thena/shared/tcga_lung_jpeg/test/*/*',
    
    'subjectID_tcgastadher2_train': '/nfs/thena/shared/tcga_stad_jpeg/train/fold*/*/*',
    'subjectID_tcgastadher2_test': '/nfs/thena/shared/tcga_stad_jpeg/test/*/*'
    }

FOLDNUM = 5


for dataset, path_glob in ROOT.items():
    data = {}    

    if 'train' in dataset:
        path_list_category = {}
        subjectNUM_category = {}
        category_list = [s.split('/')[-1] for s in glob.glob(path_glob[:-2])]
        for category in category_list:
            path_list_category[category] = glob.glob(os.path.join(path_glob[:-4], category, '*'))
            random.shuffle(path_list_category[category])
            random.shuffle(path_list_category[category])
            random.shuffle(path_list_category[category])
            subjectNUM_category[category] = float(len(path_list_category[category]))/float(FOLDNUM)

        for fold in range(1,(FOLDNUM+1)):
            for category in category_list:
                for path_category in path_list_category[category][ round((fold-1)*subjectNUM_category[category]) : round(fold*subjectNUM_category[category]) ]:
                    data[path_category.split('/')[-1]] = os.path.join('train', f'fold{fold}', path_category.split('/')[-2], path_category.split('/')[-1])

    elif 'test' in dataset:
        path_list = glob.glob(path_glob)
        random.shuffle(path_list)
        random.shuffle(path_list)
        random.shuffle(path_list)

        for path_subjectID in glob.glob(path_glob):
            data[path_subjectID.split('/')[-1]] = os.path.join(*path_subjectID.split('/')[-3:])
    else:
        raise ValueError('Something wrong')

    with open(f"preprocess/{dataset}.pkl", "wb") as fp:
        pickle.dump(data, fp)



############# 제대로 됬는지 확인 #############
# for dataset in ROOT.keys():
#     with open(f'preprocess/{dataset}.pkl', 'rb') as fp:
#         tr= pickle.load(fp)

#     print(f'{dataset} length: {len(tr.keys())}')