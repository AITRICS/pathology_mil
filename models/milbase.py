import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import torch.optim as optim
from utils import CosineAnnealingWarmUpSingle, CosineAnnealingWarmUpRestarts
from einops import rearrange


class Classifier_instance(nn.Module):
    def __init__(self, dim_in, dim_out, layer_depth=0, num_head=1):
        super(Classifier_instance, self).__init__()
        if layer_depth == 2:
            _temp = []
            for i in range(num_head):
                _temp.append(nn.Sequential(
                                        nn.Linear(dim_in, dim_in),
                                        nn.LayerNorm(dim_in),
                                        nn.ReLU(),
                                        # nn.Dropout(0.5),
                                        nn.Linear(dim_in, dim_out)
                                    ))
                self.fc = nn.ModuleList(_temp)

        elif layer_depth == 1:    
            _temp = []
            for i in range(num_head):
                _temp.append(nn.Linear(dim_in, dim_out))
            self.fc = nn.ModuleList(_temp)

        elif layer_depth == 0:
            self.fc = nn.Identity()

        
        if num_head == 1:
            self.forward = self.forward_singlehead
            if layer_depth > 0:
                self.fc = self.fc[0]
        else:
            self.forward = self.forward_multihead

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
    
    def forward_multihead(self, x):
        """
        x: num_patch x fs

        return: Length_sequence x fs x Head_num
        """
        return torch.stack([_fc(x) for _fc in self.fc], dim=2)
         
    
    def forward_singlehead(self, x):
        return self.fc(x)



class MilBase(nn.Module):
    """
    1) define train_instance:
                            (None)
                            semisup1
                            semisup2
                            intrainstance_divdis
                            interinstance_vc
                            interinstance_cosine
                            intrainstance_vc
                            intrainstance_cosine
    2) set self.negative_centroid
    3) set self.instance_classifier

    """
    def __init__(self, args, criterion=None, ma_dim_in=2048, ic_dim_in:int=2048):
        """
        ma --> mil aggregator
        ic --> instance classifier
        """
        super().__init__()
        self.args = args
        self.optimizer = {}
        self.scheduler = {}
        self.ma_dim_in = ma_dim_in
        self.alpha = args.alpha
        self.beta = args.beta
        self.dataset = args.dataset
        
        if 'instance' in args.train_instance:
            if args.ic_depth == 0:
                ic_dim_out = ic_dim_in
                assert args.ic_num_head == 1
            else:
                ic_dim_out = 128
        else:
            ic_dim_out = args.num_classes

        if criterion is not None:
            self.criterion_bag = criterion
        else:
            self.criterion_bag = nn.BCEWithLogitsLoss()        
        self.sigmoid = nn.Sigmoid()

        if args.train_instance != 'None':
            self.instance_classifier = Classifier_instance(dim_in=ic_dim_in, dim_out=ic_dim_out, layer_depth=args.ic_depth, num_head=args.ic_num_head)
            # self.set_negative_centroid(args=args, dim_negative_centroid=ic_dim_out)
       
        if args.train_instance=='intrainstance_cosine':
            self.mask_diag = 1.0-torch.eye(args.ic_num_head, requires_grad=False).unsqueeze(0).cuda()
        

    def set_negative_centroid(self, args, dim_negative_centroid):

        if (args.train_instance == 'interinstance_vc') or (args.train_instance == 'intrainstance_vc'):
            self.negative_centroid = nn.Parameter(torch.zeros((1, dim_negative_centroid), requires_grad=True).cuda())
        elif (args.train_instance == 'interinstance_cosine') or (args.train_instance=='intrainstance_cosine'):
            self.negative_centroid = nn.Parameter(torch.zeros(dim_negative_centroid, requires_grad=True).cuda())

        if hasattr(self, 'negative_centroid'):
            if self.args.optimizer_nc == 'adam':
                self.optimizer['negative_centroid'] = optim.Adam(params=[self.negative_centroid], lr=self.args.lr_center, weight_decay=0.0)
            elif self.args.optimizer_nc == 'adamw':
                self.optimizer['negative_centroid'] = optim.AdamW(params=[self.negative_centroid], lr=self.args.lr_center, weight_decay=0.0)
            elif self.args.optimizer_nc == 'sgd':
                self.optimizer['negative_centroid'] = optim.SGD(params=[self.negative_centroid], lr=self.args.lr_center, weight_decay=0.0)

        # args.num_step = len(loader_train)
        if self.args.scheduler_centroid == 'single':
            self.scheduler['negative_centroid'] = CosineAnnealingWarmUpSingle(self.optimizer['negative_centroid'], max_lr=self.args.lr_center, epochs=self.args.epochs, steps_per_epoch=self.args.num_step)
        elif self.args.scheduler_centroid == 'multi':
            self.scheduler['negative_centroid'] = CosineAnnealingWarmUpRestarts(self.optimizer['negative_centroid'], eta_max=self.args.lr_center, step_total=self.args.epochs*self.args.num_step)

    def forward(self, x: torch.Tensor):
        """
        <INPUT>
        x: #bags x #instances x #dims => encoded patches

        <OUTPUT> <- None if unnecessary
        Dict
        logit_bag: #bags x #class
        logit_instance: #instances x fs (x Head_num)
    
        """
        pass
        
    @torch.no_grad()
    def infer(self, x: torch.Tensor):
        """
        <INPUT>
        x: #bags x #instances x #dims => encoded patches

        <OUTPUT> <- None if unnecessary
        Dict
        prob_bag: #bags x #class
        prob_instance: #bags x #instances x #class
        """
        logit_dict = self.forward(x)

        prob_bag = self.sigmoid(logit_dict['bag'])
        prob_instance = self.sigmoid(logit_dict['instance']) if 'instance' in logit_dict.keys() else None
        return prob_bag, prob_instance
    
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

        <설명>
        logit_dict['bag']: 
        """

        logit_dict = self.forward(X)
        print(logit_dict['bag'].shape)
        loss_bag = self.criterion_bag(logit_dict['bag'], Y)

        # if 'instance' in logit_dict.keys():
        if self.args.train_instance == 'None':
            return loss_bag
        else:
            loss_instance = getattr(self, self.args.train_instance)(logit_dict['instance'], int(Y[0, 0]))
            return loss_bag + loss_instance
            

    def update(self, X, Y, epoch):
        """
        X: #bags x #instances x #dims => encoded patches
        Y: #bags x #classes  ==========> slide-level label

        <OUTPUT>
        None
        """
        for _optim in self.optimizer.values():
            _optim.zero_grad()
        loss = self.calculate_objective(X, Y)
        loss.backward()
        
        
        for _optim in self.optimizer.values():
            _optim.step()

        
        for _scheduler in self.scheduler.values():
            _scheduler.step()
    
    def unlabeled_weight(self, epoch):
        alpha = 0.0
        if epoch > self.T1:
            alpha = (epoch-self.T1) / (self.T2-self.T1)*self.af
            if epoch > self.T2:
                alpha = self.af
        return alpha
    
    def semisup1(self, p: torch.Tensor, target=0):
        """ 
        <INPUT>
            p: Length_sequence x Head_num
            target: wsi label. Either 0 or 1

        <return>
            loss: scalar
        """
        if self.dataset == 'CAMELYON16':
            n_instances = p.size(0)
            if target == 0:
                return self.criterion(p.sigmoid(), torch.zeros_like(p, device=p.get_device()))
            elif target == 1:
                # pseudo labeling
                with torch.no_grad():
                    pseudo_prob = p.sigmoid()
                    computed_instances_labels = torch.zeros(pseudo_prob.shape, device=p.get_device()).float()
                    mask_instances_labels = torch.zeros(pseudo_prob.shape, device=p.get_device()).float()
                    
                    _, topk_idx = torch.topk(pseudo_prob, k=int(self.alpha*n_instances), dim=0)
                    computed_instances_labels[topk_idx] = 1.
                    mask_instances_labels[topk_idx] = 1.
                    if self.beta > 0.:
                        _, bottomk_idx = torch.topk(pseudo_prob, k=int(self.beta*n_instances), largest=False, dim=0)
                        computed_instances_labels[bottomk_idx] = 0.
                        mask_instances_labels[bottomk_idx] = 1.
                
                # calculate pseudo labeled loss
                pl_loss = (self.criterion(pseudo_prob, computed_instances_labels) * mask_instances_labels).sum() / mask_instances_labels.sum()
                
                # weighting labeled loss            
                # pl_loss = self.unlabeled_weight()*pl_loss
                                
                return pl_loss
        elif self.dataset == 'tcga_lung':
            n_instances = p.size(0)
            p = p.squeeze()
            if target == 0:
                with torch.no_grad():
                    type0_pred = p[:, 0]
                    type1_pred = p[:, 1]
                    type0_prob = type0_pred.sigmoid()
                    type1_prob = type1_pred.sigmoid()
                    labeled_loss = self.criterion(type1_prob, torch.zeros_like(type1_prob, device=type1_prob.get_device()))
                    
                    computed_instances_labels = torch.zeros(type0_prob.shape, device=p.get_device()).float()
                    mask_instances_labels = torch.zeros(type0_prob.shape, device=p.get_device()).float()
                    
                    _, topk_idx = torch.topk(type0_prob, k=int(self.alpha*n_instances), dim=0)
                    computed_instances_labels[topk_idx] = 1.
                    mask_instances_labels[topk_idx] = 1.
                    if self.beta > 0.:
                        _, bottomk_idx = torch.topk(type0_prob, k=int(self.beta*n_instances), largest=False, dim=0)
                        computed_instances_labels[bottomk_idx] = 0.
                        mask_instances_labels[bottomk_idx] = 1.
                
                # calculate pseudo labeled loss
                pl_loss = (self.criterion(type0_prob, computed_instances_labels) * mask_instances_labels).sum() / mask_instances_labels.sum()
                
                # weighting labeled loss            
                # pl_loss = self.unlabeled_weight()*pl_loss
                
                total_loss = labeled_loss + pl_loss
                                
                return total_loss
            
            elif target == 1:
                # pseudo labeling
                with torch.no_grad():
                    type0_pred = p[:, 0]
                    type1_pred = p[:, 1]
                    type0_prob = type0_pred.sigmoid()
                    type1_prob = type1_pred.sigmoid()
                    labeled_loss = self.criterion(type0_prob, torch.zeros_like(type0_prob, device=type0_prob.get_device()))
                    
                    computed_instances_labels = torch.zeros(type1_pred.shape, device=p.get_device()).float()
                    mask_instances_labels = torch.zeros(type1_pred.shape, device=p.get_device()).float()
                    
                    _, topk_idx = torch.topk(type1_pred, k=int(self.alpha*n_instances), dim=0)
                    computed_instances_labels[topk_idx] = 1.
                    mask_instances_labels[topk_idx] = 1.
                    if self.beta > 0.:
                        _, bottomk_idx = torch.topk(type1_pred, k=int(self.beta*n_instances), largest=False, dim=0)
                        computed_instances_labels[bottomk_idx] = 0.
                        mask_instances_labels[bottomk_idx] = 1.
                
                # calculate pseudo labeled loss
                pl_loss = (self.criterion(type1_pred, computed_instances_labels) * mask_instances_labels).sum() / mask_instances_labels.sum()
                
                # weighting labeled loss            
                # pl_loss = self.unlabeled_weight()*pl_loss
                                
                total_loss = labeled_loss + pl_loss
                                
                return total_loss
            
            
    def semisup2(self, p: torch.Tensor, target=0):
        """ 
        <INPUT>
            p: Length_sequence x Head_num
            target: wsi label. Either 0 or 1

        <return>
            loss: scalar
        """
        pass


    def intrainstance_divdis(self, p: torch.Tensor, target=0):
        """
        Length_sequence x Head_num
        notation : H - head , D - num_class, B - sequence length(본 논문의 batch size)
        """    
        if target == 0:
            return self.criterion_bag(p, torch.zeros_like(p, device=p.get_device()))
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
    

    def interinstance_vc(self, p: torch.Tensor, target=0):
        """
        1) center(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs
        """

        ls, fs = p.shape
        _p = p - p.mean(dim=0)
        # loss_variance = torch.mean(F.relu(1.0 - torch.sqrt(p.var(dim=0) + 0.00001)))
        _p = F.normalize(_p, dim=1, eps=1e-8)
        cov = (_p.T @ _p) / (ls - 1.0)
        loss = self.args.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss

        # print(f'==================================')
        # print(f'cov: {self.off_diagonal(cov).pow_(2).sum().div(fs) }')

        if target == 0:
            _negative_centroid = p - self.negative_centroid.expand(ls, -1) # _negative_centroid : Length_sequence x fs
            loss += self.args.weight_agree * torch.mean(torch.sqrt(_negative_centroid.var(dim=0) + 0.00001)) # var loss
            # print(f'var: {self.args.weight_agree * torch.mean(torch.sqrt(_negative_centroid.var(dim=0) + 0.00001))}')
            # print(f'center location: {self.negative_centroid[0,:5]}')
        elif target == 1:
            loss += self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree - torch.sqrt(_p.var(dim=0) + 0.00001))) # variance
            # print(f'variance: {self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree - torch.sqrt(_p.var(dim=0) + 0.00001)))}')
            # print(f'max variance: {torch.amax(_p.var(dim=0))}')

        return loss

    def interinstance_cosine(self, p: torch.Tensor, target=0):
        """
        1) cosine(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs
        """
        ls, fs = p.shape
        p_n = F.normalize(p, dim=1, eps=1e-8)
        _p = p - p.mean(dim=0)
        # loss_variance = torch.mean(F.relu(1.0 - torch.sqrt(p.var(dim=0) + 0.00001)))
        _p = F.normalize(_p, dim=1, eps=1e-8)
        cov = (_p.T @ _p) / (ls - 1.0)
        loss = self.args.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss
        
        # print(f'==================================')
        # print(f'cov: {self.off_diagonal(cov).pow_(2).sum().div(fs) }')

        if target == 0:
            _negative_centroid = F.normalize(self.negative_centroid, dim=0, eps=1e-8) # _negative_centroid : fs
            p = F.normalize(p, dim=1, eps=1e-8)
            loss -= self.args.weight_agree * torch.mean(p @ _negative_centroid) # cosine loss
            # print(f'cosine - neg: {self.args.weight_agree * torch.mean(p @ _negative_centroid)}')
            # print(f'cosine location: {self.negative_centroid[:5]}')
        elif target == 1:
            loss += self.args.weight_disagree * torch.sum((p_n @ p_n.T).fill_diagonal_(0))/(ls*(ls-1.0))
            # print(f'cosine - pos: {self.args.weight_disagree * torch.sum((p_n @ p_n.T).fill_diagonal_(0))/(ls*(ls-1.0))}')
            # print(f'max variance: {torch.amax(p_n.var(dim=0))}')

        return loss
    
    def intrainstance_vc(self, p: torch.Tensor, target=0):
        """
        1) cosine(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs x Head_num
        """
        
        ls, fs, hn = p.shape
        p_whiten_head = p - p.mean(dim=2, keepdim=True) # Length_sequence x fs x Head_num
        p_whiten = p - p.mean(dim=(0,2), keepdim=True) # Length_sequence x fs x Head_num
        # p_whiten_merged = torch.transpose(p_whiten, 1, 2).view(ls*hn, fs) # (Length_sequence x Head_num) x fs
        p_whiten_merged = rearrange(torch.transpose(p_whiten, 1, 2).contiguous(), "l h f -> (l h) f") # p_whiten_merged: (Length_sequence x Head_num) x fs
        p_whiten_merged = F.normalize(p_whiten_merged, dim=1, eps=1e-8)
        
        cov = (p_whiten_merged.T @ p_whiten_merged) / ((ls*hn) - 1.0)
        loss = self.args.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss
        # print(f'==================================')
        # print(f'cov: {self.args.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs))}')
        if target == 0:
            _negative_centroid = self.negative_centroid.expand(ls*hn, fs) # _negative_centroid : (Length_sequence x Head_num) x fs 
            loss += self.args.weight_agree * torch.mean(torch.sqrt((p_whiten_merged - _negative_centroid).var(dim=0, keepdim=False) + 0.00001))
            # print(f'var_neg: {self.args.weight_agree * torch.mean(torch.sqrt((p_whiten_merged - _negative_centroid).var(dim=0, keepdim=False) + 0.00001))}')
            # print(f'negative center location: {self.negative_centroid[0,:5]}')
            
        elif target == 1:
            loss += self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree - torch.sqrt(p_whiten_head.var(dim=2) + 0.00001))) # standard deviation
            # print(f'variance: {self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree - torch.sqrt(p_whiten_head.var(dim=2) + 0.00001)))}')
            # print(f'max variance: {torch.amax(p_whiten_head.var(dim=2))}')
        # print(f'weight: {[f[0].weight[0,0].item() for f in self.instance_classifier.fc]}')
        return loss

    def intrainstance_cosine(self, p: torch.Tensor, target=0):
        """
        1) cosine(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs x Head_num
        """
        
        ls, fs, hn = p.shape
        p_t = torch.transpose(p, 1, 2) # Length_sequence x Head_num x fs
        p_whiten = p - p.mean(dim=(0,2), keepdim=True) # Length_sequence x fs x Head_num
        # p_whiten_merged = torch.transpose(p_whiten, 1, 2).view(ls*hn, fs) # (Length_sequence x Head_num) x fs
        p_whiten_merged = rearrange(torch.transpose(p_whiten, 1, 2).contiguous(), "l h f -> (l h) f") # p_whiten_merged: (Length_sequence x Head_num) x fs
        p_whiten_merged = F.normalize(p_whiten_merged, dim=1, eps=1e-8)
        cov = (p_whiten_merged.T @ p_whiten_merged) / ((ls*hn) - 1.0)
        loss = self.args.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs)) # covariance loss
        # print(f'==================================')
        # print(f'cov: {self.args.weight_cov*(self.off_diagonal(cov).pow_(2).sum().div(fs))}')

        if target == 0:
            _negative_centroid = F.normalize(self.negative_centroid, dim=0, eps=1e-8) # _negative_centroid : fs
            p = F.normalize(p_t, dim=2, eps=1e-8) # p: Length_sequence x Head_num x fs
            loss -= self.args.weight_agree * torch.mean(p @ _negative_centroid) # Length_sequence x Head_num
            # print(f'cosine loss: {self.args.weight_agree * torch.mean(p @ _negative_centroid)}')
            # print(f'center location: {self.negative_centroid[:5]}')
        elif target == 1:
            loss += self.args.weight_disagree * torch.sum(torch.bmm(F.normalize(p_t, dim=2, eps=1e-8), F.normalize(p, dim=1, eps=1e-8)) * self.mask_diag)/(ls*hn*(hn-1.0))
            # print(f'variance: {self.args.weight_disagree * torch.sum(torch.bmm(F.normalize(p_t, dim=2, eps=1e-8), F.normalize(p, dim=1, eps=1e-8)) * self.mask_diag)/(ls*hn*(hn-1.0))}')
            # print(f'max variance: {torch.amax(F.normalize(p, dim=1, eps=1e-8).var(dim=2))}')

        return loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




