import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import torch.optim as optim
from utils import CosineAnnealingWarmUpSingle, CosineAnnealingWarmUpRestarts
from einops import rearrange


class Classifier_instance(nn.Module):
    def __init__(self, n_channels, num_head, aux_head=0):
        super(Classifier_instance, self).__init__()
        if aux_head == 2:
            self.fc = nn.Sequential(
                                        nn.Linear(n_channels, n_channels),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(n_channels, num_head)
                                    )
        elif aux_head == 1:    
            self.fc = nn.Linear(n_channels, num_head)
        elif aux_head == 0:
            self.fc = nn.Identity()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
    
    def forward(self, x):
        return self.fc(x)


class MilBase(nn.Module):
    def __init__(self, args, optimizer=None, criterion=None, scheduler=None, dim_in:int=2048, dim_latent: int=512, dim_out=1, aux_loss=None, aux_head=None, weight_cov=None, auxloss_weight=None):
        super().__init__()
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler

        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # if scheduler is not None:
        #     self.scheduler = scheduler
        # else:
        #     if args.scheduler == 'single':
        #         self.scheduler = CosineAnnealingWarmUpSingle(self.optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=args.num_step)
        #     elif args.scheduler == 'multi':
        #         self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, eta_max=args.lr, step_total=args.epochs*args.num_step)

        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.aux_loss = aux_loss
        self.aux_head = aux_head
        self.weight_cov = weight_cov
        self.auxloss_weight = auxloss_weight
        
        if args.mil_model == 'dsmil':     
            channels= dim_in
        else :
            channels = dim_latent
        if aux_loss =='loss_divdis':
            self.num_head = 2
        else :
            self.num_head = 128
        self.instance_classifier = Classifier_instance(channels, num_head=self.num_head, aux_head=aux_head)    
        self.scaler = torch.cuda.amp.GradScaler()
        self.sigmoid = nn.Sigmoid()
        
        if aux_loss != 'None':
            self.criterion_aux = getattr(self, aux_loss)
        else:
            assert aux_head == 0

    def loss_dbat(self, p: torch.Tensor, target=0):
        """
        D-BAT
        p: Length_sequence x Head_num(2)
        """
        
        if target == 0:
            loss = self.criterion(p, torch.zeros_like(p, device=p.get_device()))            
        elif target == 1:
            p = torch.sigmoid(p)
            loss = -torch.log(p[:,0]*(1.0-p[:,1]) + (1.0-p[:,0])*p[:,1] +  1e-7).mean()
        # (- torch.log(p_1_s[i] * (1-p_2_s[i]) + p_2_s[i] * (1-p_1_s[i]) +  1e-7)).mean()
        return loss

    def loss_jsd(self, p: torch.Tensor, target=0):
        """
        Jensen-Shannon Divergence
        p: Length_sequence x Head_num(2)
        """
        if target == 0:
            return self.criterion(p, torch.zeros_like(p, device=p.get_device()))
        elif target == 1:
            p = torch.sigmoid(p)
            return 0.5*(F.kl_div(p[:,0], p[:,1], reduction='batchmean', log_target=False) + F.kl_div(p[:,1], p[:,0], reduction='batchmean', log_target=False))
    
    def loss_vc(self, p: torch.Tensor, target=None):
        """
        Variance + Covariance
 
        p: Length_sequence x fs
        """
        ls, fs = p.shape
        p = p - p.mean(dim=0)
        loss_variance = torch.mean(F.relu(1.0 - torch.sqrt(p.var(dim=0) + 0.00001)))
        cov = (p.T @ p) / (ls - 1.0)
        loss_covariance = self.off_diagonal(cov).pow_(2).sum().div(fs)
        return loss_variance + (self.weight_cov * loss_covariance)


    def loss_divdis(self, p: torch.Tensor, target=0):
        """
        Length_sequence x Head_num
        notation : H - head , D - num_class, B - sequence length(본 논문의 batch size)
        """    
        if target == 0:
            return self.criterion(p, torch.zeros_like(p, device=p.get_device()))
        elif target ==1:
            p = p.sigmoid().unsqueeze(-1) 
            p = torch.cat([p, 1 - p], dim=-1) # B, H, D
            marginal_p = p.mean(dim=0)  # H, D 
            marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)  # H, H, D, D
            marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2

            joint_p = torch.einsum("bhd,bge->bhgde", p, p).mean(dim=0)  # H, H, D, D
            joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2

            # Compute pairwise mutual information = KL(P_XY | P_X x P_Y)
            # Equivalent to: F.kl_div(marginal_p.log(), joint_p, reduction="none")
            kl_computed = joint_p * (joint_p.log() - marginal_p.log()) 
            kl_computed = kl_computed.sum(dim=-1)
            kl_grid = rearrange(kl_computed, "(h g) -> h g", h=p.shape[1])
            repulsion_grid = -kl_grid
            repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
            repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)] # 처음에 non_zero인게 없음
            if torch.sum(repulsions) ==0:
                repulsion_loss = 1e-7
            else : 
                repulsion_loss = -repulsions.mean()

            return repulsion_loss
        
    def set_optimizer(self):

        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

        if self.scheduler is None:
            if self.args.scheduler == 'single':
                self.scheduler = CosineAnnealingWarmUpSingle(self.optimizer, max_lr=self.args.lr, epochs=self.args.epochs, steps_per_epoch=self.args.num_step)
            elif self.args.scheduler == 'multi':
                self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, eta_max=self.args.lr, step_total=self.args.epochs*self.args.num_step)
        
        if self.aux_head != 0:
            self.optimizer2 = torch.optim.Adam(self.instance_classifier.parameters(), lr=self.args.lr,  weight_decay=1e-4) ## psuedo bag
            self.scheduler2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer2, '[100]', gamma=0.2)  
            
    def forward(self, x: torch.Tensor):
        """
        <INPUT>
        x: #bags x #instances x #dims => encoded patches

        <OUTPUT> <- None if unnecessary
        logit_bag: #bags x #class
        logit_instance: #bags x #instances x #class
        """
        pass

    def infer(self, x: torch.Tensor):
        """
        <INPUT>
        x: #bags x #instances x #dims => encoded patches

        <OUTPUT> <- None if unnecessary
        prob_bag: #bags x #class
        prob_instance: #bags x #instances x #class
        """
        logit_bag, logit_instance,_ = self.forward(x)

        prob_bag = self.sigmoid(logit_bag)
        prob_instance = self.sigmoid(logit_instance) if logit_instance is not None else None
        return prob_bag, prob_instance
    
    def calculate_objective(self, X, Y):
        """
        <INPUT>
        X: #bags x #instances x #dims => encoded patches
        Y: #bags x #classes  ==========> slide-level label

        <OUTPUT>
        loss: scalar
        """
        pass

    def update(self, X, Y):
        """
        X: #bags x #instances x #dims => encoded patches
        Y: #bags x #classes  ==========> slide-level label

        <OUTPUT>
        None
        """
        
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss1, loss2 = self.calculate_objective(X, Y)
            
        if self.aux_loss != 'None':
            loss = loss1 + (self.auxloss_weight * loss2)
        else:
            loss = loss1

        # print("loss: ", loss)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

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


# INPUT: #bags x #instances x #dims
# OUTPUT: #bags x #classes

from models.transformer.encoder import TransformerEncoder

class MilTransformer(MilBase):

    # def __init__(self, encoder=None, num_heads=8, num_layers=3, share_proj=False, balance_param=math.log(39./252.),  **kwargs):
    def __init__(self, if_learn_instance, pseudo_prob_threshold, encoder=None, share_proj=True, **kwargs):
        super().__init__(**kwargs)

        if encoder == None:
            self.encoder = nn.Sequential(
                # nn.Dropout(p=0.3),
                nn.Linear(self.dim_in, self.dim_latent),
                nn.ReLU(),
            )
        elif encoder == 'resnet':
            assert self.dim_latent == 512
            from torchvision.models import resnet18, ResNet18_Weights
            self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.encoder.fc = nn.Identity()
        else:
            self.encoder = encoder()
        
        # self.pool = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim_latent, nhead=num_heads), num_layers=num_layers, enable_nested_tensor=True)    
        # dim = 512, head = 8, layer = 3
        # self.pool = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim_latent, nhead=num_heads, batch_first=False), num_layers=num_layers)    
        # self.pool = TransformerEncoder(d_input = self.dim_latent, 
        #          n_layers = num_layers, 
        #          n_head = num_heads,
        #          d_model = self.dim_latent, 
        #          d_ff = self.dim_latent * 4, 
        #          use_pe = False)
        self.pool = TransformerEncoder(d_input = self.dim_latent, 
                 n_layers = 2, 
                 n_head = 2,
                 d_model = self.dim_latent, 
                 d_ff = self.dim_latent*2, 
                 sr_ratio = 8,
                 use_pe = False,
                 layerwise_shuffle = self.args.layerwise_shuffle)    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim_latent, requires_grad=True))
        
        self.score_bag = nn.Linear(self.dim_latent, self.dim_out, bias=True)
        self.score_instance = nn.Linear(self.dim_latent, self.dim_out, bias=True)
        self.share_proj = share_proj
        self.criterion = nn.BCEWithLogitsLoss()
        self.cnt_over_thresh = 0
        self.if_learn_instance = if_learn_instance
        self.pseudo_prob_threshold = pseudo_prob_threshold
        
        self.set_optimizer()
        
    def forward(self, x: torch.Tensor):
        x = torch.cat([self.cls_token, self.encoder(x)], dim=1) # #slide x (1 + #patches) x dim_latent

        x = self.pool(x)

        if self.share_proj:
            # #slide x #dim_out,  # #slide x #patches x #dim_out
            return self.score_bag(x[:, 0, :]), self.score_bag(x[:, 1:, :])
        else:
            # #slide x #dim_out,  # #slide x #patches x #dim_out
            return self.score_bag(x[:, 0, :]), self.score_instance(x[:, 1:, :])
    
    def calculate_objective(self, X, Y):
        # X: #bag x (1 + #patches) x dim_latent
        # Y: #bag x #classes
        logit_bag, logit_instance = self.forward(X)
        # logit_bag: #bag x #classes
        # logit_instance: #bag x #instance x #classes
        loss = self.criterion(logit_bag + self.args.balance_param, Y)

        if self.if_learn_instance:
            if Y == 0:
                # loss_instance = torchvision.ops.sigmoid_focal_loss(logit_instance+math.log(39./252.), target.unsqueeze(1).repeat(logit_instance.size(0), logit_instance.size(1), logit_instance.size(2)), reduction='mean')
                loss_instance = self.criterion(logit_instance+self.args.balance_param, Y.unsqueeze(1).expand(logit_instance.size(0), logit_instance.size(1), logit_instance.size(2)))
            else:
                #slide x #patches x num_class
                # logit_instance = logit_instance.squeeze(0)
                # #patches x num_class
                prob_instance = F.sigmoid(logit_instance)
                pseudo_label_positive = torch.zeros_like(prob_instance, device=X.device)
                pseudo_label_positive[prob_instance>self.pseudo_prob_threshold] = 1.0
                
                mask = pseudo_label_positive.detach().clone()
                mask[prob_instance<(1.0-self.pseudo_prob_threshold)]=1.0
                if (torch.sum(pseudo_label_positive) < 1):
                    # loss_instance = torchvision.ops.sigmoid_focal_loss(torch.max(logit_instance)+math.log(39./252.), torch.ones([], device=args.device))
                    loss_instance = self.criterion(torch.max(logit_instance) + self.args.balance_param, torch.ones([], device=X.device))
                else:
                    # loss_instance = torch.sum(mask * torchvision.ops.sigmoid_focal_loss(logit_instance+math.log(39./252.), pseudo_label_positive))/torch.sum(mask)
                    loss_instance = torch.sum(mask * F.binary_cross_entropy_with_logits(logit_instance + self.args.balance_param, pseudo_label_positive, reduction='none'))/torch.sum(mask)
                    self.cnt_over_thresh+=1

            loss += loss_instance

        return loss


    
# class MilTransformer_(nn.Module):

#     def __init__(self, encoder=None, dim_in:int=2048, dim_latent: int= 512, dim_out = 1, num_heads=8, num_layers=3, share_proj=False):
#         super().__init__()

#         if encoder == None:
#             self.encoder = nn.Sequential(
#                 # nn.Dropout(p=0.3),
#                 nn.Linear(dim_in, dim_latent),
#                 nn.ReLU(),
#             )
#         elif encoder == 'resnet':
#             assert dim_latent == 512
#             from torchvision.models import resnet18, ResNet18_Weights
#             self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
#             self.encoder.fc = nn.Identity()
#         else:
#             self.encoder = encoder()
        
#         # self.pool = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim_latent, nhead=num_heads), num_layers=num_layers, enable_nested_tensor=True)    
#         self.pool = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim_latent, nhead=num_heads, batch_first=True), num_layers=num_layers)    
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_latent, requires_grad=True))

#         self.score_bag = nn.Linear(dim_latent, dim_out, bias=True)
#         self.score_instance = nn.Linear(dim_latent, dim_out, bias=True)
#         self.share_proj = share_proj

#     def forward(self, x: torch.Tensor):

#         x = torch.cat([self.cls_token, self.encoder(x)], dim=1)
#         # x --> #slide x (1 + #patches) x dim_latent

#         x = self.pool(x)
#         # x --> #slide x (1 + #patches) x dim_latent

#         if self.share_proj:
#             return self.score_bag(x[:, 0, :]), self.score_bag(x[:, 1:, :])
#         else:
#             return self.score_bag(x[:, 0, :]), self.score_instance(x[:, 1:, :])    
    


def milmax(encoder=None, dim_in:int=2048, dim_latent: int= 512, dim_out = 1):
    return MilCustom(encoder=encoder, dim_in=dim_in, dim_latent=dim_latent, dim_out=dim_out, pool=nn.AdaptiveMaxPool1d((1)))



def milmean(encoder=None, dim_in:int=2048, dim_latent: int= 512, dim_out = 1):
    return MilCustom(encoder=encoder, dim_in=dim_in, dim_latent=dim_latent, dim_out=dim_out, pool=nn.AdaptiveAvgPool1d((1)))