import os
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from .milbase import Classifier_instance
from einops import rearrange
from itertools import chain
from .milbase import MilBase


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        # for ii in range(numLayer_Res):
        #     self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x

class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred
    
def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps    


class Dtfd(MilBase):
    def __init__(self,args, ma_dim_in:int=2048):
        super().__init__(args=args, ma_dim_in=ma_dim_in, ic_dim_in=512)
        
        mDim=512
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.numGroup = 5
        self.instance_per_group = 1
        self.classifier = Classifier_1fc(mDim, args.num_classes, 0)
        self.attention = Attention_Gated(mDim)
        self.dimReduction = DimReduction(n_channels=ma_dim_in, m_dim=mDim, numLayer_Res=0)
        self.UClassifier = Attention_with_Classifier(L=mDim, num_cls=args.num_classes, droprate=0)
        self.distill = 'AFS'
        self.grad_clipping = 5
        self.device = args.device
        self.sigmoid = nn.Sigmoid()
        
        if args.train_instance == 'None':
            self.optimizer['mil_model'] = torch.optim.Adam(list(self.classifier.parameters())+list(self.attention.parameters())+
                                                            list(self.dimReduction.parameters())+list(self.UClassifier.parameters()),
                                                            lr=args.lr,  weight_decay=1e-4)

        else:
            self.optimizer['mil_model'] = torch.optim.Adam(list(self.classifier.parameters())+list(self.attention.parameters())+
                                                            list(self.dimReduction.parameters())+list(self.UClassifier.parameters())+
                                                            list(self.instance_classifier.parameters()), lr=args.lr,  weight_decay=1e-4)
                
        self.scheduler['mil_model'] = torch.optim.lr_scheduler.MultiStepLR(self.optimizer['mil_model'], '[100]', gamma=0.2)
        
    def first_tier(self, x: torch.Tensor) :
        slide_sub_preds = []
        slide_pseudo_feat = []
        instance_pseudo_feat = []
        
        feat_index = list(range(x.shape[1]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        
        for tindex in index_chunk_list:
            subFeat_tensor = torch.index_select(x.squeeze(0), dim=0, index=torch.LongTensor(tindex).to(self.device))
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA) # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  ### 1 x cls
            instance_pseudo_feat.append(tattFeats)
            slide_sub_preds.append(tPredict)
            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls
            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:self.instance_per_group].long()
            topk_idx_min = sort_idx[-self.instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if self.distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif self.distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat) 
            elif self.distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)
                
        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x args.num_classes
        instance_pseudo_feat = torch.cat(instance_pseudo_feat, dim=0) ### num_patch x fs
        
        return slide_pseudo_feat, slide_sub_preds, instance_pseudo_feat
    
    def forward(self, x: torch.Tensor):

        feat_pseudo_bag, logit_pseudo_bag, feat_instances = self.first_tier(x) ### numGroup x fs  ,   numGroup x cls

        logit_bag = self.UClassifier(feat_pseudo_bag)

        if self.args.train_instance != 'None':
            logit_instances = self.instance_classifier(feat_instances)
            # logit_bag: #bags x args.num_classes     logit_instances: #instances x ic_dim_out (x Head_num)
            # 아직 확인 못함
            return {'bag': logit_bag, 'pseudo_bag': logit_pseudo_bag, 'instance': logit_instances}
        else:       
            return {'bag': logit_bag, 'pseudo_bag': logit_pseudo_bag}
               

    def calculate_objective(self, X, Y):
        """
        <STYLE>
        1) add losses
        2) no fp16 but fp32 as usual

        <INPUT>
        X: #bags x #instances x #dims => encoded patches
        Y: #bags x #classes  ==========> slide-level label

        <OUTPUT>
        loss: scalar
        """

        logit_dict = self.forward(X)
        slide_sub_labels = Y.expand(self.numGroup, -1).to(self.device) # batchsize must be 1
        loss0 = self.criterion(logit_dict['pseudo_bag'], slide_sub_labels).mean()
        loss1 = self.criterion(logit_dict['bag'], Y).mean()
        
        if self.args.train_instance == 'None':
            return loss0 + loss1
        else:            
            loss2 = getattr(self, self.args.train_instance)(logit_dict['instance'], int(Y[0, 0]))
            return loss0 + loss1 + loss2
    
    def update(self, X, Y):
        """
        X: #bags x #instances x #dims => encoded patches
        Y: #bags x #classes  ==========> slide-level label        
        """
        for _optim in self.optimizer.values():
            _optim.zero_grad()
        loss = self.calculate_objective(X, Y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.grad_clipping)   
        torch.nn.utils.clip_grad_norm_(self.UClassifier.parameters(), self.grad_clipping)

        for _optim in self.optimizer.values():
            _optim.step()
        
        for _scheduler in self.scheduler.values():
            _scheduler.step()
