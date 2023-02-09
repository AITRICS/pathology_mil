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

