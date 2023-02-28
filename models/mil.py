import torch
import torch.nn as nn


class MilBase(nn.Module):

    def __init__(self, encoder=None, dim_in:int=2048, dim_latent: int= 512, dim_out = 1, pool = nn.AdaptiveMaxPool1d((1))):
        super().__init__()

        if encoder == None:
            self.encoder = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(dim_in, dim_latent),
                nn.ReLU(),
            )
        else:
            self.encoder = encoder

        self.pool = pool
        self.score = nn.Linear(dim_latent, dim_out, bias=True)
    
    def forward(self, x: torch.Tensor):
        
        x = self.encoder(x)

        # x --> #slide x #patches x dim_latent
        x = self.score(x)

        # x --> #slide x #patches x dim_out
        x = self.pool(torch.transpose(x,1,2)).squeeze(2)

        # x --> #slide x dim_out
        return x

class MilTransformer(MilBase):

    def __init__(self, dim_in:int=2048, dim_latent: int= 512, dim_out = 1, num_heads=8, num_layers=3, num_class=1, **kwargs):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(dim_in, dim_latent),
                nn.ReLU(),
            )
        
        self.pool = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim_latent, nhead=num_heads), num_layers=num_layers, enable_nested_tensor=True, mask_check=True)    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_latent))

        self.score_bag = nn.Linear(dim_latent, num_class, bias=True)
        self.score_instance = nn.Linear(dim_latent, num_class, bias=True)


    def forward(self, x: torch.Tensor):

        x = torch.cat([self.cls_token, self.encoder(x)], dim=1)
        # x --> #slide x (1 + #patches) x dim_latent

        x = self.pool(x)
        # x --> #slide x (1 + #patches) x dim_latent

        return self.score_bag(x[:, 0:1, :]), self.score_instance(x[:, 1:, :])

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