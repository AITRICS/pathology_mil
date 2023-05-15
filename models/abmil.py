import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import torch.optim as optim
from utils import CosineAnnealingWarmUpSingle, CosineAnnealingWarmUpRestarts
from .mil import MilBase

class Attention(MilBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.L = self.dim_latent
        self.D = 128
        self.K = self.dim_out

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        self.encoder = nn.Sequential(
            nn.Linear(self.dim_in, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L, 1)
        )
        
        if kwargs['optimizer'] is not None:
            self.optimizer = kwargs['optimizer']
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=kwargs['lr'], betas=(0.9, 0.999), weight_decay=10e-5)

        # self.set_optimizer()
        

    def forward(self, x):
        # INPUT: #bags x #instances x #dims
        # OUTPUT: #bags x #classes
        # x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        H = self.encoder(x)  # BxNxL
# H: seq(=-1) x self.L,    seq=K
        A = self.attention(H)  # BxNxK
        # A = self.attention(x)  
        A = torch.transpose(A, 2, 1)  # BxKxN
        A = F.softmax(A, dim=2)  # softmax over N

        # M = torch.mm(A, H)  # KxL
        # A: BxKxN
        # H: BxNxL
        M = torch.matmul(A, H)  # BxKxL

        logit_bag = self.classifier(M).squeeze(2) # BxK        
        # Y_hat = torch.sign(F.relu(Y_logit)).float()

        # return Y_prob, Y_hat, A
        # return F.sigmoid(Y_logit), Y_logit, Y_hat
        return logit_bag, None

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, _, Y_hat = self.forward(X)
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    #     return error, Y_hat

    def calculate_objective(self, X, Y):
        # Y = Y.float()
        # Y_prob, _, A = self.forward(X)
        logit_bag, _ = self.forward(X)
        loss = self.criterion(logit_bag, Y)
        # Y_prob = F.sigmoid(logit_bag)
        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        # return neg_log_likelihood, A
        return loss


class GatedAttention(MilBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.L = self.dim_latent
        self.D = 128
        self.K = self.dim_out

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        self.encoder = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_latent),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)  

        self.classifier = nn.Sequential(
            # nn.Linear(self.L*self.K, 1),
            nn.Linear(self.L, 1),
            # nn.Sigmoid()
        )
        
        if kwargs['optimizer'] is not None:
            self.optimizer = kwargs['optimizer']
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=kwargs['lr'], betas=(0.9, 0.999), weight_decay=10e-5)

        # self.set_optimizer()        
        
    def forward(self, x):
        # INPUT: #bags x #instances x #dims
        # OUTPUT: #bags x #classes
        # x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.encoder(H)  # NxL
        H = self.encoder(x)  # BxNxL

        A_V = self.attention_V(H)  # BxNxD
        A_U = self.attention_U(H)  # BxNxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # BxNxK
        A = torch.transpose(A, 2, 1)  # BxKxN
        A = F.softmax(A, dim=2)  # softmax over N

        # A: BxKxN
        # H: BxNxL
        M = torch.matmul(A, H)  # BxKxL

        logit_bag = self.classifier(M).squeeze(2) # BxK
        # Y_hat = torch.sign(F.relu(Y_logit)).float()

        # return F.sigmoid(Y_logit), Y_logit, Y_hat
        return logit_bag, None

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, _, Y_hat = self.forward(X)
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

    #     return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        logit_bag, _ = self.forward(X)
        loss = self.criterion(logit_bag, Y)
        return loss
        # Y_prob = F.sigmoid(logit_bag)
        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        # return neg_log_likelihood
