import torch
import torch.nn as nn


class MilBase(nn.Module):

    def __init__(self, encoder=None, dim_in:int=2048, dim_latent: int= 512, dim_out = 1, pool = nn.AdaptiveMaxPool1d((1))):
        super().__init__()

        if encoder == None:
            self.encoder = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(dim_in, dim_latent),
                # nn.LayerNorm(dim_latent),
                nn.ReLU(),
                # nn.Dropout(p=0.3),
                # nn.Linear(dim_latent, dim_latent),
                # # nn.LayerNorm(dim_latent),
                # nn.ReLU(),
            )
        else:
            self.encoder = encoder

        self.pool = pool
        self.score = nn.Linear(dim_latent, dim_out, bias=True)
    
    def forward(self, x: torch.Tensor):
        
        x = self.encoder(x)

        # x --> #slide x #patches x dim_in
        x = self.score(x)

        # x --> #slide x #patches x dim_out
        x = self.pool(torch.transpose(x,1,2)).squeeze(2)

        # x --> #slide x dim_out
        return x

# class MilMax(MilBase):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.score = nn.Sequential(nn.Linear(dim_in, dim_latent, bias=False),
#                                             nn.BatchNorm1d(dim_latent),
#                                             nn.ReLU(inplace=True),
#                                             nn.Linear(dim_latent, 1, bias=False))
    
#     def forward(, x: torch.Tensor):
        