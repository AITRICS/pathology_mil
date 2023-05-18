import glob
import os
from PIL import Image
import numpy as np
import torch
import torchvision

# 수정 포인트1
PATH_ROOT = '/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224/train'
# 수정 포인트2
path_list = glob.glob(os.path.join(PATH_ROOT, '*', '*', '*', '*.jpeg'))
# transform = T.Compose([
#         T.ToTensor()
#     ])
### mean
totensor = torchvision.transforms.ToTensor()
mean_v = torch.tensor([0., 0., 0.])
for path in path_list:
    img = totensor(Image.open(path))
    try:
        assert (img.shape[0] == 3) and (img.shape[1] == 224) and (img.shape[2] == 224) and (len(img.shape) == 3)
    except:
        raise ValueError(f'Invalid image property: {path}\nshape: {img.shape}')
    mean_v += torch.mean(img, axis=(1, 2))

mean_v /= len(path_list)

### stddev
mean_v_expanded = torch.unsqueeze(torch.unsqueeze(mean_v, 1), 2)
stddev = torch.tensor([0., 0., 0.])
## all image must be in same shape (width, height)
for path in path_list:
    stddev += torch.mean(torch.square(totensor(Image.open(path)) - mean_v_expanded), dim=(1, 2))

stddev /= len(path_list)
stddev = torch.square(stddev)

print(f'mean: {mean_v}\nstddev: {stddev}')