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


class rnn_single(nn.Module):
    
    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims

        self.fc1 = nn.Linear(512, ndims)
        self.fc2 = nn.Linear(ndims, ndims)

        self.fc3 = nn.Linear(ndims, 2)

        self.activation = nn.ReLU()

    def forward(self, input, state):
        input = self.fc1(input) 
        state = self.fc2(state)
        state = self.activation(state+input) 
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)        