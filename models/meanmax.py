import torch
import torch.nn as nn
from .milbase import MilBase


class MilCustom(MilBase):

    # def __init__(self, args, optimizer=None, criterion=None, scheduler=None, encoder=None, dim_in:int=2048, dim_latent: int= 512, dim_out = 1, pool = nn.AdaptiveMaxPool1d((1))):
    def __init__(self, encoder=None, pool = nn.AdaptiveMaxPool1d((1)), **kwargs):
        super().__init__(**kwargs)

        if encoder == None:
            self.encoder = nn.Sequential(
                # nn.Dropout(p=0.3),
                nn.Linear(self.dim_in, self.dim_latent),
                nn.ReLU(),
            )
        else:
            self.encoder = encoder

        self.pool = pool
        self.score = nn.Linear(self.dim_latent, self.dim_out, bias=True)
        

    def forward(self, x: torch.Tensor):
        
        x = self.encoder(x) # #slide x #patches x dim_latent

        x = self.score(x) # #slide x #patches x dim_out

        logit_bag = self.pool(torch.transpose(x,1,2)).squeeze(2) # #slide x #dim_out
        # Y_hat = torch.sign(F.relu(Y_logit)).float()

        return logit_bag, None

    def calculate_objective(self, X, Y):
        logit_bag, _ = self.forward(X)
        loss = self.criterion(logit_bag, Y)

        return loss    


    


def milmax(encoder=None, dim_in:int=2048, dim_latent: int= 512, dim_out = 1):
    return MilCustom(encoder=encoder, dim_in=dim_in, dim_latent=dim_latent, dim_out=dim_out, pool=nn.AdaptiveMaxPool1d((1)))



def milmean(encoder=None, dim_in:int=2048, dim_latent: int= 512, dim_out = 1):
    return MilCustom(encoder=encoder, dim_in=dim_in, dim_latent=dim_latent, dim_out=dim_out, pool=nn.AdaptiveAvgPool1d((1)))
