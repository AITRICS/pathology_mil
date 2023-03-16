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

    def __init__(self,dim_in=2048, dim_latent=512, dim_out=1, mil_mode="att") :

        super().__init__()
        self.mil_mode = mil_mode
        self.attention = nn.Sequential()
        self.transformer = None  # type: Optional[nn.Module]
        self.dim_in = dim_in
        self.dim_latent = dim_latent 
        self.num_cls = dim_out
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.encoder = nn.Sequential(
            # nn.Dropout(p=0.3),
            nn.Linear(dim_in, dim_latent),
            nn.ReLU(),
        )

        if self.mil_mode in ["mean", "max"]:
            pass
        elif self.mil_mode == "att":
            self.attention = nn.Sequential(nn.Linear(self.dim_latent, 2048), nn.Tanh(), nn.Linear(2048, 1))

        elif self.mil_mode in ["att_trans"]:
            transformer = nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=8, dropout=0)
            self.transformer = nn.TransformerEncoder(transformer, num_layers=4)
            for i, (name, layer) in enumerate(self.transformer.named_children()):
                if isinstance(layer, nn.ReLU):
                    setattr(self.transformer, name, nn.ReLU(inplace=False))
            self.attention = nn.Sequential(nn.Linear(self.dim_latent, 2048), nn.Tanh(), nn.Linear(2048, 1))
            

        else:
            raise ValueError("Unsupported mil_mode: " + str(mil_mode))

        self.myfc = nn.Linear(self.dim_latent, self.num_cls)
        self.scorefc= nn.Linear(self.dim_latent, self.num_cls)

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

            x = x.permute(1, 0, 2) #permute 후 (n,b,dim)
            x = self.transformer(x)
            x = x.permute(1, 0, 2) # (b,n,dim)

            a = self.attention(x) # att shape (b,n,cls)
            a = torch.softmax(a, dim=1) # att shape (b,n,cls)
            x = torch.sum(x * a, dim=1) # x*a shape (b,n,dim) , sum 후 (b,dim)
            # myfc = nn.Linear(dim, num_classes)
            x = self.myfc(x)

        else:
            raise ValueError("Wrong model mode" + str(self.mil_mode))

        return x

    def forward(self, x: torch.Tensor, no_head: bool = False) -> torch.Tensor:
        # input: #bags x #instances x #dims
        x = self.encoder(x)
        bag_logit = self.calc_head(x)
        return bag_logit, None

    def calculate_objective(self, X, Y):
        bag_logit, _ = self.forward(X)
        bag_loss = self.criterion(bag_logit,Y)
        
        return bag_loss