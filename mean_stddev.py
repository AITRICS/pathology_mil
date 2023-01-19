import glob
import os
from PIL import Image
import numpy as np

# 수정 포인트1
PATH_ROOT = '/mnt/aitrics_ext/ext01/shared/hazel/cam_jpeg_train/train'
# 수정 포인트2
path_list = glob.glob(os.path.join(PATH_ROOT, '*', '*', '*.jpeg'))

### mean
mean_v = np.array([0., 0., 0.])
for path in path_list:
    img = np.array(Image.open(path))
    try:
        assert (img.shape[0] == 384) and (img.shape[1] == 384) and (img.shape[2] == 3) and (len(img.shape) == 3)
    except:
        raise ValueError(f'Invalid image property: {path}\nshape: {img.shape}')
    mean_v += np.mean(img[64:-64, 64:-64, :], axis=(0, 1))

mean_v /= len(path_list)

### stddev
mean_v_expanded = np.expand_dims(np.expand_dims(mean_v, 0), 0)
stddev = np.array([0., 0., 0.])
## all image must be in same shape (width, height)
for path in path_list:
    stddev += np.mean(np.square(np.array(Image.open(path))[64:-64, 64:-64, :] - mean_v_expanded), axis=(0, 1))

stddev /= len(path_list)
stddev = np.sqrt(stddev)

print(f'mean: {mean_v}\nstddev: {stddev}')