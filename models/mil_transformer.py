import torch
import torch.nn as nn
from models.transformer.encoder import TransformerEncoder
from .milbase import MilBase
# INPUT: #bags x #instances x #dims
# OUTPUT: #bags x #classes


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
