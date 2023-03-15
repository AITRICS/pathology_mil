import os
import torch
import torch.nn as nn
import random
import numpy as np

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
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
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
    
class dtfd_tier1(nn.Module):
    def __init__(self,encoder=None, dim_in:int=2048, dim_latent=512, dim_out=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_latent = dim_latent
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.numGroup = 5 
        self.instance_per_group = 1 # 확인하기
        self.classifier = Classifier_1fc(self.dim_latent, self.dim_out, 0)
        self.attention = Attention_Gated(self.dim_latent)
        self.dimReduction = DimReduction(self.dim_in, self.dim_latent, numLayer_Res=0)
        self.UClassifier = Attention_with_Classifier(L=self.dim_latent, num_cls=self.dim_out, droprate=0)
        self.distill = 'AFS'
        self.grad_clipping = 5
        
        
    def first_tier(self, x: torch.Tensor, y) :
        slide_sub_preds = []
        slide_pseudo_feat = []
        slide_sub_labels = []
        
        feat_index = list(range(x.shape[1]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        
        for tindex in index_chunk_list:
            slide_sub_labels.append(y)
            subFeat_tensor = torch.index_select(x, dim=0, index=torch.LongTensor(tindex))
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA) # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  ### 1 x cls
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
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x 2
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
        
        loss0 = self.criterion(slide_sub_preds, slide_sub_labels).mean()
        optimizer0.zero_grad()
        loss0.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.grad_clipping)   
        
           
    def forward(self, x: torch.Tensor):
        logit_bag = self.UClassifier(x)
        
        return logit_bag, None 
           
    def calculate_objective(self, X, Y):
        slide_pseudo_feat = self.first_tier(X,Y)
        logit_bag, _ = self.forward(slide_pseudo_feat)
        return self.criterion(logit_bag, Y).mean()
