# Whole Slide Image (WSI) classification using Multiple Instance Learning

* [MONAI Tutorial](https://github.com/Project-MONAI/tutorials/tree/main/pathology/multiple_instance_learning)
* [DDP tutorial](https://pytorch.org/docs/stable/distributed.elastic.html)

## Installation

1. Install Conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
source ~/.bashrc
```
2. Create and activate Conda environment  (mil as an env name)
```
conda create -n mil python=3.8
conda activate mil
```
3. Install PyTorch
* CUDA 11.3
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
* CUDA 11.6
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
4. Install MONAI (mil as an env name)
```
~/anaconda3/envs/mil/bin/pip install 'monai[all]'
```

5. Install OpenCV
```
~/anaconda3/envs/mil/bin/pip install opencv-python

```


## Run Command (under construction)
* **possible** DDP run command (--nproc_per_node -> number of gpus per node):
```
torchrun --standalone --nnodes=1 --nproc_per_node=$NUMBER_OF_GPUS_PER_NODE main.py
```

## Dependencies
* For CUDA 10.2
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
```

* For CUDA 11.1
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```
