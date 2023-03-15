import torch
import torch.nn as nn
from typing import Dict, Optional, Union, cast


#customed / backbone deleted
class MonaiMil(nn.Module):
    """
    Multiple Instance Learning (MIL) model, with a backbone classification model.
    Currently, it only works for 2D images, a typical use case is for classification of the
    digital pathology whole slide images. The expected shape of input data is `[B, N, C, H, W]`,
    where `B` is the batch_size of PyTorch Dataloader and `N` is the number of instances
    extracted from every original image in the batch. A tutorial example is available at:
    https://github.com/Project-MONAI/tutorials/tree/master/pathology/multiple_instance_learning.

    Args:
        num_classes: number of output classes.
        mil_mode: MIL algorithm, available values (Defaults to ``"att"``):

            - ``"mean"`` - average features from all instances, equivalent to pure CNN (non MIL).
            - ``"max"`` - retain only the instance with the max probability for loss calculation.
            - ``"att"`` - attention based MIL https://arxiv.org/abs/1802.04712.
            - ``"att_trans"`` - transformer MIL https://arxiv.org/abs/2111.01556.


        pretrained: init backbone with pretrained weights, defaults to ``True``.
        backbone: Backbone classifier CNN (either ``None``, a ``nn.Module`` that returns features,
            or a string name of a torchvision model).
            Defaults to ``None``, in which case ResNet50 is used.
        backbone_num_features: Number of output features of the backbone CNN
            Defaults to ``None`` (necessary only when using a custom backbone)
        trans_blocks: number of the blocks in `TransformEncoder` layer.
        trans_dropout: dropout rate in `TransformEncoder` layer.

    """

    def __init__(self,dim_in=2048, dim_latent=512, dim_out=1, mil_mode="att_trans") :

        super().__init__()
        self.mil_mode = mil_mode
        self.attention = nn.Sequential()
        self.transformer = None  # type: Optional[nn.Module]
        self.dim_in = dim_in
        self.dim_latent = dim_latent 
        self.num_cls = dim_out

        if self.mil_mode in ["mean", "max"]:
            pass
        elif self.mil_mode == "att":
            self.attention = nn.Sequential(nn.Linear(self.dim_in, 2048), nn.Tanh(), nn.Linear(2048, 1))

        elif self.mil_mode in ["att_trans"]:
            transformer = nn.TransformerEncoderLayer(d_model=self.dim_in, nhead=8, dropout=trans_dropout)
            self.transformer = nn.TransformerEncoder(transformer, num_layers=trans_blocks)
            for i, (name, layer) in enumerate(self.transformer.named_children()):
                if isinstance(layer, nn.ReLU):
                    setattr(self.transformer, name, nn.ReLU(inplace=False))
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))
            

        else:
            raise ValueError("Unsupported mil_mode: " + str(mil_mode))

        self.myfc = nn.Linear(nfc, num_classes)
        self.scorefc= nn.Linear(nfc, num_classes)

    def calc_head(self, x: torch.Tensor) -> torch.Tensor:

        sh = x.shape

        if self.mil_mode == "mean":
            x = self.myfc(x)
            x = torch.mean(x, dim=1)

        elif self.mil_mode == "max":
            x = self.myfc(x)
            x, _ = torch.max(x, dim=1)

        elif self.mil_mode == "att":

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            x = self.myfc(x)

        elif self.mil_mode == "att_trans" and self.transformer is not None:

            x = x.permute(1, 0, 2) #permute 후 (n,b,512)
            x = self.transformer(x)
            x = x.permute(1, 0, 2) # (b,n,512)

            a = self.attention(x) # att shape (b,n,1)
            a = torch.softmax(a, dim=1) # att shape (b,n,1)
            x = torch.sum(x * a, dim=1) # x*a shape (b,n,512) , sum 후 (b,512)
            # myfc = nn.Linear(512, num_classes)
            x = self.myfc(x)

        else:
            raise ValueError("Wrong model mode" + str(self.mil_mode))

        return x

    def forward(self, x: torch.Tensor, no_head: bool = False) -> torch.Tensor:

        sh = x.shape
        x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])

        x = self.net(x)
        x = x.reshape(sh[0], sh[1], -1)
        
        if not no_head:
            x = self.calc_head(x)

        return x