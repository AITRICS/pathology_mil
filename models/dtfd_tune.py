import os
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from .mil import Classifier_instance
from einops import rearrange
from itertools import chain

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
    
class Dtfd_tune(nn.Module):
    def __init__(self, args, optimizer=None, criterion=None, scheduler=None,encoder=None, dim_in:int=2048, dim_latent=512, dim_out=1,
                 aux_loss = None, num_head = 0, layernum_head=0, weight_agree=1.0, weight_disagree=1.0, weight_cov=1.0, stddev_disagree=1.0):
        super().__init__()

        if ((aux_loss == 'loss_dbat') or (aux_loss == 'loss_divdis') or (aux_loss == 'loss_jsd')):
            fs = dim_out
        else:
            fs = 128

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_latent = dim_latent
        self.criterion = nn.BCEWithLogitsLoss() # 원래 CE 였는디..
        self.numGroup = 5
        self.instance_per_group = 1 # 확인하기
        self.classifier = Classifier_1fc(self.dim_latent, self.dim_out, 0)
        self.attention = Attention_Gated(self.dim_latent)
        self.dimReduction = DimReduction(self.dim_in, self.dim_latent, numLayer_Res=0)
        self.UClassifier = Attention_with_Classifier(L=self.dim_latent, num_cls=self.dim_out, droprate=0)
        self.distill = 'AFS'
        self.grad_clipping = 5
        self.device = args.device
        self.sigmoid = nn.Sigmoid()
        self.instance_classifier = Classifier_instance(dim_latent, fs=fs, layernum_head=layernum_head, num_head=num_head)
        self.aux_loss = aux_loss
        self.num_head = num_head
        
        if aux_loss != 'None':
            self.criterion_aux = getattr(self, aux_loss)
        else:
            assert layernum_head == 0
        
        if (aux_loss == 'loss_center') or (aux_loss == 'loss_var') or (aux_loss == 'loss_div_vc'):
            self.representative_vector = nn.Parameter(torch.ones((1, fs)).cuda())
        elif (aux_loss == 'loss_contrastive') or (aux_loss=='loss_div_contrastive'):
            self.representative_vector = nn.Parameter(torch.ones((fs)).cuda())
        elif aux_loss == 'loss_center_vc':
            self.representative_vector = nn.Parameter(torch.ones((1, fs, 1)).cuda())
            assert num_head >=3
        elif aux_loss == 'loss_cosine_vc':
            self.representative_vector = nn.Parameter(torch.ones((fs)).cuda())
            assert num_head >=3

        if aux_loss=='loss_div_contrastive':
            self.mask_diag = 1.0-torch.eye(num_head, requires_grad=False).unsqueeze(0).cuda()
            

        self.weight_agree = weight_agree
        self.weight_disagree = weight_disagree
        self.weight_cov = weight_cov
        self.stddev_disagree = stddev_disagree
        
        # update
        self.optimizer0 = torch.optim.Adam(list(self.classifier.parameters())+list(self.attention.parameters())+list(self.dimReduction.parameters()), lr=args.lr,  weight_decay=1e-4)
        self.optimizer1 = torch.optim.Adam(self.UClassifier.parameters(), lr=args.lr,  weight_decay=1e-4) ## psuedo bag
        if layernum_head != 0:
            self.optimizer2 = torch.optim.Adam(list(self.instance_classifier.parameters())+[self.representative_vector], lr=args.lr,  weight_decay=1e-4) ## psuedo bag

        self.scheduler0 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer0, '[100]', gamma=0.2)
        self.scheduler1 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer1, '[100]', gamma=0.2)
        if num_head != 0:
            self.scheduler2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer2, '[100]', gamma=0.2)       
        
    def first_tier(self, x: torch.Tensor):
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
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x cls
        instance_pseudo_feat = torch.cat(instance_pseudo_feat, dim=0) ### num_patch x fs
        
        return slide_pseudo_feat, slide_sub_preds, instance_pseudo_feat
    
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
        return self.weight_cov * loss

    def loss_jsd(self, p: torch.Tensor, target=0):
        """
        Jensen-Shannon Divergence
        p: Length_sequence x Head_num(2)
        """
        if target == 0:
            return self.criterion(p, torch.zeros_like(p, device=p.get_device()))
        elif target == 1:
            p = torch.sigmoid(p)
            return self.weight_cov*0.5*(F.kl_div(p[:,0], p[:,1], reduction='batchmean', log_target=False) + F.kl_div(p[:,1], p[:,0], reduction='batchmean', log_target=False))
    
    def loss_center(self, p: torch.Tensor, target=0):
        """
        1) center(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs
        """

        ls, fs = p.shape
        _p = p - p.mean(dim=0)
        # loss_variance = torch.mean(F.relu(1.0 - torch.sqrt(p.var(dim=0) + 0.00001)))
        cov = (_p.T @ _p) / (ls - 1.0)
        loss = self.off_diagonal(cov).pow_(2).sum().div(fs) # covariance loss

        print(f'==================================')
        print(f'cov: {self.off_diagonal(cov).pow_(2).sum().div(fs) }')

        if target == 0:
            _representative_vector = self.representative_vector.expand(ls, -1) # _representative_vector : Length_sequence x fs
            loss += self.weight_agree * torch.mean(torch.pow(p - _representative_vector, 2).sum(dim=1, keepdim=False)) # center loss
            print(f'center: {self.weight_agree * torch.mean(torch.pow(p - _representative_vector, 2).sum(dim=1, keepdim=False))}')
            print(f'center location: {self.representative_vector[0,:5]}')
        elif target == 1:            
            loss += self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(_p.var(dim=0) + 0.00001))) # variance
            print(f'variance: {self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(_p.var(dim=0) + 0.00001)))}')
            print(f'max variance: {torch.amax(_p.var(dim=0))}')

        return loss

    def loss_var(self, p: torch.Tensor, target=0):
        """
        1) center(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs
        """

        ls, fs = p.shape
        _p = p - p.mean(dim=0)
        # loss_variance = torch.mean(F.relu(1.0 - torch.sqrt(p.var(dim=0) + 0.00001)))
        cov = (_p.T @ _p) / (ls - 1.0)
        loss = self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss

        # print(f'==================================')
        # print(f'cov: {self.off_diagonal(cov).pow_(2).sum().div(fs) }')

        if target == 0:
            _representative_vector = p - self.representative_vector.expand(ls, -1) # _representative_vector : Length_sequence x fs
            loss += self.weight_agree * torch.mean(torch.sqrt(_representative_vector.var(dim=0) + 0.00001)) # var loss
            # print(f'var: {self.weight_agree * torch.mean(torch.sqrt(_representative_vector.var(dim=0) + 0.00001))}')
            # print(f'center location: {self.representative_vector[0,:5]}')
        elif target == 1:
            loss += self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(_p.var(dim=0) + 0.00001))) # variance
            # print(f'variance: {self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(_p.var(dim=0) + 0.00001)))}')
            # print(f'max variance: {torch.amax(_p.var(dim=0))}')

        return loss
    
    def loss_contrastive(self, p: torch.Tensor, target=0):
        """
        1) cosine(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs
        """

        ls, fs = p.shape
        p_n = F.normalize(p, dim=1, eps=1e-8)
        _p = p - p.mean(dim=0)
        # loss_variance = torch.mean(F.relu(1.0 - torch.sqrt(p.var(dim=0) + 0.00001)))
        cov = (_p.T @ _p) / (ls - 1.0)
        loss = self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss
        
        # print(f'==================================')
        # print(f'cov: {self.off_diagonal(cov).pow_(2).sum().div(fs) }')

        if target == 0:
            _representative_vector = F.normalize(self.representative_vector, dim=0, eps=1e-8) # _representative_vector : fs
            p = F.normalize(p, dim=1, eps=1e-8)
            loss -= self.weight_agree * torch.mean(p @ _representative_vector) # cosine loss
            
            # print(f'cosine neg: {self.weight_agree * torch.mean(torch.pow(p - _representative_vector, 2).sum(dim=1, keepdim=False))}')
            # print(f'cosine location: {self.representative_vector[:5]}')
        elif target == 1:
            loss += self.weight_disagree * torch.sum((p_n @ p_n.T).fill_diagonal_(0))/(ls*(ls-1.0))
            # print(f'cosine - pos: {self.weight_disagree * torch.sum((p_n @ p_n.T).fill_diagonal_(0))/(ls*(ls-1.0))}')
            # print(f'max variance: {torch.amax(_p.var(dim=0))}')

        return loss
        
    
    def loss_center_vc(self, p: torch.Tensor, target=0):
        """
        1) cosine(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs x Head_num
        """

        ls, fs, hn = p.shape
        p_whiten = p - p.mean(dim=(0,2), keepdim=True) # Length_sequence x fs x Head_num
        # p_whiten_merged = torch.transpose(p_whiten, 1, 2).view(ls*hn, fs) # (Length_sequence x Head_num) x fs
        p_whiten_merged = rearrange(torch.transpose(p_whiten, 1, 2).contiguous(), "l h f -> (l h) f") # p_whiten_merged: (Length_sequence x Head_num) x fs
        
        cov = (p_whiten_merged.T @ p_whiten_merged) / ((ls*hn) - 1.0)
        loss = self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss
        print(f'==================================')
        print(f'cov: {self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs))}')
        if target == 0:
            _representative_vector = self.representative_vector.expand(ls, fs, hn) # _representative_vector : Length_sequence x fs x Head_num
            loss += self.weight_agree * torch.mean(torch.pow(p - _representative_vector, 2).sum(dim=1, keepdim=False)) # center loss
            print(f'center: {self.weight_agree * torch.mean(torch.pow(p - _representative_vector, 2).sum(dim=1, keepdim=False))}')
            print(f'center location: {self.representative_vector[0,:5,0]}')
        elif target == 1:
            loss += self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(p_whiten.var(dim=2) + 0.00001))) # standard deviation
            print(f'variance: {self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(p_whiten.var(dim=2) + 0.00001)))}')
            print(f'max variance: {torch.amax(p_whiten.var(dim=2))}')
        
        print(f'weight: {[f.weight[0,0].item() for f in self.instance_classifier.fc]}')
        return loss
    
    def loss_div_vc(self, p: torch.Tensor, target=0):
        """
        1) cosine(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs x Head_num
        """

        ls, fs, hn = p.shape
        p_whiten_head = p - p.mean(dim=2, keepdim=True) # Length_sequence x fs x Head_num
        p_whiten = p - p.mean(dim=(0,2), keepdim=True) # Length_sequence x fs x Head_num
        # p_whiten_merged = torch.transpose(p_whiten, 1, 2).view(ls*hn, fs) # (Length_sequence x Head_num) x fs
        p_whiten_merged = rearrange(torch.transpose(p_whiten, 1, 2).contiguous(), "l h f -> (l h) f") # p_whiten_merged: (Length_sequence x Head_num) x fs
        
        cov = (p_whiten_merged.T @ p_whiten_merged) / ((ls*hn) - 1.0)
        loss = self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss
        # print(f'==================================')
        # print(f'cov: {self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs))}')
        if target == 0:
            _representative_vector = self.representative_vector.expand(ls*hn, fs) # _representative_vector : (Length_sequence x Head_num) x fs 
            loss += self.weight_agree * torch.mean(torch.sqrt((p_whiten_merged - _representative_vector).var(dim=0, keepdim=False) + 0.00001))
            # print(f'var_neg: {self.weight_agree * torch.mean(torch.sqrt((p_whiten_merged - _representative_vector).var(dim=0, keepdim=False) + 0.00001))}')
            # print(f'negative center location: {self.representative_vector[0,:5]}')
            
        elif target == 1:
            loss += self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(p_whiten_head.var(dim=2) + 0.00001))) # standard deviation
            # print(f'variance: {self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(p_whiten_head.var(dim=2) + 0.00001)))}')
            # print(f'max variance: {torch.amax(p_whiten_head.var(dim=2))}')
        
        # print(f'weight: {[f.weight[0,0].item() for f in self.instance_classifier.fc]}')
        return loss

    def loss_div_contrastive(self, p: torch.Tensor, target=0):
        """
        1) cosine(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs x Head_num
        """

        ls, fs, hn = p.shape
        p_t = torch.transpose(p, 1, 2) # Length_sequence x Head_num x fs
        p_whiten = p - p.mean(dim=(0,2), keepdim=True) # Length_sequence x fs x Head_num
        # p_whiten_merged = torch.transpose(p_whiten, 1, 2).view(ls*hn, fs) # (Length_sequence x Head_num) x fs
        p_whiten_merged = rearrange(torch.transpose(p_whiten, 1, 2).contiguous(), "l h f -> (l h) f") # p_whiten_merged: (Length_sequence x Head_num) x fs
        
        cov = (p_whiten_merged.T @ p_whiten_merged) / ((ls*hn) - 1.0)
        loss = self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss
        # print(f'==================================')
        # print(f'cov: {self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs))}')

        if target == 0:
            _representative_vector = F.normalize(self.representative_vector, dim=0, eps=1e-8) # _representative_vector : fs
            p = F.normalize(p_t, dim=2, eps=1e-8) # p: Length_sequence x Head_num x fs
            loss -= self.weight_agree * torch.mean(p @ _representative_vector) # Length_sequence x Head_num
            # print(f'cosine loss: {self.weight_agree * torch.mean(p @ _representative_vector)}')
            # print(f'center location: {self.representative_vector[:5]}')
        elif target == 1:
            loss += self.weight_disagree * torch.sum(torch.bmm(F.normalize(p_t, dim=2, eps=1e-8), F.normalize(p, dim=1, eps=1e-8)) * self.mask_diag)/(ls*hn*(hn-1.0))
            # loss += self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(p_whiten.var(dim=2) + 0.00001))) # standard deviation
            # print(f'variance: {self.weight_disagree * torch.sum(torch.bmm(F.normalize(p_t, dim=2, eps=1e-8), F.normalize(p, dim=1, eps=1e-8)) * self.mask_diag)/(ls*hn*(hn-1.0))}')
            # print(f'max variance: {torch.amax(p_whiten.var(dim=2))}')

        return loss
    


    def loss_cosine_vc(self, p: torch.Tensor, target=0):
        """
        1) cosine(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs x Head_num
        """

        ls, fs, hn = p.shape
        p_whiten = p - p.mean(dim=(0,2), keepdim=True) # Length_sequence x fs x Head_num
        # p_whiten_merged = torch.transpose(p_whiten, 1, 2).view(ls*hn, fs) # (Length_sequence x Head_num) x fs
        p_whiten_merged = rearrange(torch.transpose(p_whiten, 1, 2).contiguous(), "l h f -> (l h) f") # p_whiten_merged: (Length_sequence x Head_num) x fs
        
        cov = (p_whiten_merged.T @ p_whiten_merged) / ((ls*hn) - 1.0)
        loss = self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss
        # print(f'==================================')
        # print(f'cov: {self.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs))}')

        if target == 0:
            _representative_vector = F.normalize(self.representative_vector, dim=0, eps=1e-8) # _representative_vector : fs
            p = F.normalize(torch.transpose(p, 1, 2), dim=2, eps=1e-8) # p: Length_sequence x Head_num x fs
            loss -= self.weight_agree * torch.mean(p @ _representative_vector) # Length_sequence x Head_num
            # print(f'cosine loss: {self.weight_agree * torch.mean(p @ _representative_vector)}')
            # print(f'center location: {self.representative_vector[:5]}')
        elif target == 1:
            loss += self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(p_whiten.var(dim=2) + 0.00001))) # standard deviation
            # print(f'variance: {self.weight_disagree * torch.mean(F.relu(self.stddev_disagree - torch.sqrt(p_whiten.var(dim=2) + 0.00001)))}')
            # print(f'max variance: {torch.amax(p_whiten.var(dim=2))}')

        return loss
        

    def loss_vc(self, p: torch.Tensor, target=None):
        """
        Variance + Covariance
 
        p: Length_sequence x fs
        """
        ls, fs = p.shape
        _p = p - p.mean(dim=0)
        loss_variance = torch.mean(F.relu(1.0 - torch.sqrt(_p.var(dim=0) + 0.00001)))
        cov = (_p.T @ _p) / (ls - 1.0)
        loss_covariance = self.off_diagonal(cov).pow_(2).sum().div(fs)
        return (self.weight_disagree * loss_variance) + (self.weight_cov * loss_covariance)

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
           
    def forward(self, x: torch.Tensor):

        feat_pseudo_bag, logit_pseudo_bag, feat_instances = self.first_tier(x) ### numGroup x fs      ,      numGroup x cls

        logit_bag = self.UClassifier(feat_pseudo_bag)
        logit_instances = self.instance_classifier(feat_instances)
        # logit_bag: K(=1) x cls
        return logit_bag, logit_pseudo_bag, logit_instances 

    def calculate_objective(self, X, Y):
        # X => 1 x #instance x self.dim_in, Y => #bags x #classes
        # logit_bag: K(=1) x cls
        # logit_pseudo_bag: numGroup x cls
        logit_bag, logit_pseudo_bag, logit_instances = self.forward(X) 
        # slide_sub_labels
        slide_sub_labels = torch.ones((self.numGroup,1)).to(self.device)*Y
        # loss0 = self.criterion(logit_pseudo_bag, slide_sub_labels).mean()
        loss0 = self.criterion(logit_pseudo_bag, slide_sub_labels).mean()
        loss1 = self.criterion(logit_bag, Y).mean()
        if self.aux_loss != 'None':
            loss2 = self.criterion_aux(logit_instances, Y[0, 0])
        else:
            loss2 = None

        return loss0, loss1, loss2
    
    def update(self, X, Y, alpha=0.0):
        """
        X: #bags x #instances x #dims => encoded patches
        Y: #bags x #classes  ==========> slide-level label
        """
        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()
        with torch.cuda.amp.autocast():
            loss0, loss1, loss2 = self.calculate_objective(X, Y)
        # print(f'{loss0.item()}, {loss1.item()}, {loss2.item()}')
        loss0.backward(retain_graph=True)
        if self.aux_loss != 'None':
            loss1.backward(retain_graph=True)
            loss2.backward()
        else:
            loss1.backward()

        torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.UClassifier.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.instance_classifier.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.representative_vector, self.grad_clipping)

        self.optimizer0.step()
        self.optimizer1.step()
        if self.num_head != 0:
            self.optimizer2.step()

        self.scheduler0.step()
        self.scheduler1.step()
        if self.num_head != 0:
            self.scheduler2.step()


    def infer(self, x: torch.Tensor):
        """
        <INPUT>
        x: #bags x #instances x #dims => encoded patches

        <OUTPUT> <- None if unnecessary
        prob_bag: #bags x #class
        prob_instance: #bags x #instances x #class
        """
        logit_bag, _, _ = self.forward(x)
        # print(f'logit_bag: {logit_bag}')
        prob_bag = self.sigmoid(logit_bag)

        return prob_bag, None