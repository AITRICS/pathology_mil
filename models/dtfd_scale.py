import os
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F

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
    
class Dtfd_scale(nn.Module):
    def __init__(self,args, optimizer=None, criterion=None, scheduler=None,encoder=None, dim_in:int=2048, dim_latent=512, dim_out=1, **kwargs):
        super().__init__()
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
        self.scaler = torch.cuda.amp.GradScaler()
        
        # update
        self.optimizer0 = torch.optim.Adam(list(self.classifier.parameters())+ list(self.attention.parameters())+list(self.dimReduction.parameters()), lr=args.lr,  weight_decay=1e-4)
        self.optimizer1 = torch.optim.Adam(self.UClassifier.parameters(), lr=args.lr,  weight_decay=1e-4)
        self.scheduler0 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer0, '[100]', gamma=0.2)
        self.scheduler1 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer1, '[100]', gamma=0.2)        
        
    def first_tier(self, x: torch.Tensor) :
        slide_sub_preds = []
        slide_pseudo_feat = []
        
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
        
        return slide_pseudo_feat, slide_sub_preds
        
           
    def forward(self, x: torch.Tensor):

        feat_pseudo_bag, logit_pseudo_bag = self.first_tier(x) ### numGroup x fs      ,      numGroup x cls

        logit_bag = self.UClassifier(feat_pseudo_bag)
        # logit_bag: K(=1) x cls
        return logit_bag, logit_pseudo_bag 
           
    def calculate_objective(self, X, Y):
        # X => 1 x #instance x self.dim_in, Y => #bags x #classes
        # logit_bag: K(=1) x cls
        # logit_pseudo_bag: numGroup x cls
        logit_bag, logit_pseudo_bag = self.forward(X) 
        # slide_sub_labels
        slide_sub_labels = torch.ones((self.numGroup,1)).to(self.device)*Y
        loss0 = self.criterion(logit_pseudo_bag, slide_sub_labels).mean()
        loss1 = self.criterion(logit_bag, Y).mean()
        return loss0, loss1
    
    def update(self, X, Y):
        """
        X: #bags x #instances x #dims => encoded patches
        Y: #bags x #classes  ==========> slide-level label        
        """
        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()
        with torch.cuda.amp.autocast():
            loss0, loss1 = self.calculate_objective(X, Y)
        # loss0.backward(retain_graph=True)
        # loss1.backward()
        self.scaler.scale(loss0).backward(retain_graph=True)
        self.scaler.scale(loss1).backward()

        torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.grad_clipping)   
        torch.nn.utils.clip_grad_norm_(self.UClassifier.parameters(), self.grad_clipping)

        # self.optimizer0.step()
        # self.optimizer1.step()
        self.scaler.step(self.optimizer0)
        self.scaler.step(self.optimizer1)
        self.scaler.update()
        
        self.scheduler0.step()
        self.scheduler1.step()

    def infer(self, x: torch.Tensor):
        """
        <INPUT>
        x: #bags x #instances x #dims => encoded patches

        <OUTPUT> <- None if unnecessary
        prob_bag: #bags x #class
        prob_instance: #bags x #instances x #class
        """
        logit_bag, _ = self.forward(x)

        prob_bag = self.sigmoid(logit_bag)

        return prob_bag, None