from openslide import OpenSlide
import glob
import csv
import pickle

# FILENAME = 'CAMELYON16'
FILENAME = 'tcga_lung'
# FILENAME = 'tcga_stad_wsi'

a=[
    # '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/*/*.tiff',
    # '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/*/*/*.tiff',
    # '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/*/*/*/*.tiff',

    # '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/*/*.ndpi',
    # '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/*/*/*.ndpi',
    # '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/*/*/*/*.ndpi',

    # '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/*/*.svs',
    # '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/*/*/*.svs',
    # '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/*/*/*/*.svs',

    # '/nfs/thena/shared/gc_segmentation/with_ano_SSH_60/*.ndpi'


    # f'/nfs/thena/shared/camelyon/CAMELYON16/images/*'
    f'/nfs/thena/shared/{FILENAME}/*/*'
    ]
# with open(f'preprocess/subjectID_camelyon16_train.pkl', 'rb') as fp:
#     tr= pickle.load(fp)
# with open(f'preprocess/subjectID_camelyon16_test.pkl', 'rb') as fp:
#     tt= pickle.load(fp)

with open(f'preprocess/subjectID_tcgalung_train.pkl', 'rb') as fp:
    tr= pickle.load(fp)
with open(f'preprocess/subjectID_tcgalung_test.pkl', 'rb') as fp:
    tt= pickle.load(fp)

# with open(f'preprocess/subjectID_tcgastadher2_train.pkl', 'rb') as fp:
#     tr= pickle.load(fp)
# with open(f'preprocess/subjectID_tcgastadher2_test.pkl', 'rb') as fp:
#     tt= pickle.load(fp)

cnt = 1
with open(f'{FILENAME}.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile)
    # wr.writerow([cnt, 'path', 'level_count', 'mpp-x', 'mpp-y', 'height_0', 'width_0', 'ds_0', 'height_1', 'width_1', 'ds_1', 'height_2', 'width_2', 'ds_2'])
    wr.writerow([cnt, 'path', 'level_count', 'mpp-x', 'mpp-y', 'mpp_highest_x', 'mpp_highest_y'])
    cnt += 1

for pss in a:
    for ps in glob.glob(pss):
        subjectID = ps.split('/')[-1][:-4]
        if (subjectID in tr.keys()) or (subjectID in tt.keys()):
            img = OpenSlide(ps)
            level_count = img.level_count
            if ps[-3:] == 'tif':
                try:
                    mppx = float(10**4) / float(img.properties['tiff.XResolution'])
                    mppy = float(10**4) / float(img.properties['tiff.YResolution'])
                except:
                    mppx = img.properties['openslide.mpp-x']
                    mppy = img.properties['openslide.mpp-y']                    
            else:
                try:
                    mppx = img.properties['openslide.mpp-x']
                    mppy = img.properties['openslide.mpp-y']
                except:                
                    raise ValueError(f'something wrong !!!!!!!!!!: {subjectID}\n\nkey\n{img.properties.keys()}')
                        
            mpp_highest_x = float(mppx) * img.level_downsamples[int(img.level_count-1)]
            mpp_highest_y = float(mppy) * img.level_downsamples[int(img.level_count-1)]

                    # print(f'')
            # ds = []

            # level = img.level_count if img.level_count < 4 else 3
            # for i in range(level):
            #     ds.append( [img.properties[f'openslide.level[{i}].downsample']] )
            # ih = []
            # for i in range(level):
            #     ih.append( [img.properties[f'openslide.level[{i}].height']] )
            # iw = []
            # for i in range(level):
            #     iw.append( [img.properties[f'openslide.level[{i}].width']] )

            with open(f'{FILENAME}.csv', 'a', newline='') as csvfile:            
                wr = csv.writer(csvfile)
                tw = [cnt, ps, level_count, mppx, mppy, mpp_highest_x, mpp_highest_y]
                # for l in range(level):
                #     tw.append(ih[l][0])
                #     tw.append(iw[l][0])
                #     tw.append(ds[l][0])
                # for l in range(level):
                # for l in range(level):

                wr.writerow(tw)
                # wr.writerow([cnt, ps, mppx, mppy, int(ih[0][0]), int(ih[1][0]), int(ih[2][0]), int(iw[0][0]), int(iw[1][0]), int(iw[2][0]), int(ds[0][0]), float(ds[1][0]), float(ds[2][0]) ])
                cnt += 1
                
        
    

