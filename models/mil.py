import torch
import torch.nn as nn


class Aggregator(nn.Module):

    def __init__(self, encoder = nn.Identity(), dim_in:int=2048, dim_latent: int= 512, pool = nn.AdaptiveMaxPool1d((1))):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.score = nn.Sequential(nn.Linear(dim_in, dim_latent, bias=False),
                                            nn.BatchNorm1d(dim_latent),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(dim_latent, 1, bias=False))
    
    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.score(x)
        x = self.pool(x)
        return x
