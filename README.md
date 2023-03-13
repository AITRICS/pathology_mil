# Whole Slide Image (WSI) classification using Multiple Instance Learning

* [MONAI Tutorial](https://github.com/Project-MONAI/tutorials/tree/main/pathology/multiple_instance_learning)
* [DDP tutorial](https://pytorch.org/docs/stable/distributed.elastic.html)

## Disclaimer
This repo is tested under Python 3.8, PyTorch 1.12+cuda11.3


## Run Command (under construction)
* **possible** DDP run command (--nproc_per_node -> number of gpus per node):
```
torchrun --standalone --nnodes=1 --nproc_per_node=$NUMBER_OF_GPUS_PER_NODE main.py
```

## Installation

```
conda update -n base -c defaults conda -y
conda create -n p1131 python=3.8 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c anaconda scikit-learn -y
conda install -c conda-forge tqdm -y
~/anaconda3/envs/p1131/bin/pip install pushbullet-python

```