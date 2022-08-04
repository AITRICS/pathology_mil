# Whole Slide Image (WSI) classification using Multiple Instance Learning

* [MONAI Tutorial](https://github.com/Project-MONAI/tutorials/tree/main/pathology/multiple_instance_learning)

## Run Command (under construction)
* possible DDP run command (--nproc_per_node -> number of gpus per node):
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
