import torch
import torch.nn as nn
import torch.nn.functional as F
from .mil import milmax

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
    
    
class milrnn(nn.Module):
    def __init__(self,encoder=None, dim_in:int=2048, dim_latent=512, dim_out=1, model_path:str=''):
        super().__init__()
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.criterion = nn.CrossEntropyLoss()
        self.rnn = rnn_single(128)
        if model_path != '':
            self.model = milmax(encoder=None, dim_in=self.dim_in, dim_latent=self.dim_latent, dim_out = self.dim_out)
            state_dict = torch.load(model_path)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict['state_dict'], strict=False)
            assert missing_keys == [] and unexpected_keys == []
        else :
            ValueError('model path not found')
    
        
    def forward(self, x: torch.Tensor):
        self.model.eval()
        midfeats = self.model.encoder(x) # bs x patch x latent
        scores = self.model.score(midfeats).squeeze(-1) # bs x patch
        top_indices = torch.argsort(scores, dim=-1)[:,:10] # bs x s
        top_midfeats = torch.index_select(midfeats, dim=1, index=top_indices[0]) # assume bs ==1
        assert top_midfeats.shape == (1,10,512)
        
        rnn_bs = top_midfeats.size(0) # 원래 128인데 틀에 넣느라 1
        self.rnn.zero_grad()
        state = self.rnn.init_hidden(rnn_bs)
        for s in range(top_midfeats[0].size(1)):
            input = top_midfeats[:,:,s,:].squeeze() # bs x fs
            logit_bag, state = self.rnn(input, state) #output : bs x num_cls, state : bs x ndims
            
        return logit_bag, None
    
    def calculate_objective(self, X, Y):
        logit_bag, _ = self.forward(X)
        return self.criterion(logit_bag, Y)