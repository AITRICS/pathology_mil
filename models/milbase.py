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

        elif layer_depth == 3:
            _temp = []
            for i in range(num_head):
                _temp.append(nn.Sequential(
                                        nn.Linear(dim_in, dim_in),
                                        nn.LayerNorm(dim_in),
                                        nn.ReLU(),
                                        nn.Linear(dim_in, dim_in),
                                        nn.LayerNorm(dim_in),
                                        nn.ReLU(),
                                        nn.Linear(dim_in, dim_out)
                                    ))
                self.fc = nn.ModuleList(_temp)

        elif layer_depth == 4:
            _temp = []
            for i in range(num_head):
                _temp.append(nn.Sequential(
                                        nn.Linear(dim_in, dim_in),
                                        nn.LayerNorm(dim_in),
                                        nn.ReLU(),
                                        nn.Linear(dim_in, dim_in),
                                        nn.LayerNorm(dim_in),
                                        nn.ReLU(),
                                        nn.Linear(dim_in, dim_in),
                                        nn.LayerNorm(dim_in),
                                        nn.ReLU(),
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

        ours, divdis
        """
        return torch.stack([_fc(x) for _fc in self.fc], dim=2)
         
    
    def forward_singlehead(self, x):
        """
        x: num_patch x fs

        return: Length_sequence x fs

        baseline, semi-sups
        """
        return self.fc(x)



class MilBase(nn.Module):
    """
    1) define train_instance:
                            (None)
                            semisup1
                            semisup2
                            divdis
                            interinstance_vi
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
        self.var_pos=[]
        self.var_neg=[]
        self.init_mean = 0.5
        self.init_std = 0.1

        self.std_pos = []        
        self.std_neg = []

        # 더 이상 사용 안함
        assert 'intra' not in args.train_instance

        if args.ic_depth == 0:
            ic_dim_out = ic_dim_in
        
        if 'instance' in args.train_instance:
            assert args.ic_depth > 0
            assert args.ic_num_head == 1
            ic_dim_out = 128
            # if args.ic_depth == 0:
            #     ic_dim_out = ic_dim_in
            #     assert args.ic_num_head == 1
            # else:
            #     ic_dim_out = 128
        else:
            ic_dim_out = args.num_classes
        
        self.ic_dim_out = ic_dim_out
        self.cs = torch.nn.CosineSimilarity(dim=0)

        if criterion is not None:
            self.criterion_bag = criterion
        else:
            self.criterion_bag = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        if args.train_instance != 'None':
            self.instance_classifier = Classifier_instance(dim_in=ic_dim_in, dim_out=ic_dim_out*args.num_classes, layer_depth=args.ic_depth, num_head=args.ic_num_head)
            self.set_negative_centroid(args=args, dim_negative_centroid=ic_dim_out)
       
        if args.train_instance=='intrainstance_cosine':
            self.mask_diag = 1.0-torch.eye(args.ic_num_head, requires_grad=False).unsqueeze(0).cuda()
        

    def set_negative_centroid(self, args, dim_negative_centroid):

        if ('interinstance_vi' in args.train_instance) or (args.train_instance == 'intrainstance_vc'):
            # self.negative_centroid = nn.Parameter(torch.zeros((args.num_classes, dim_negative_centroid), requires_grad=True).cuda() + 0.1 if self.args.mil_model=='Dsmil' else 0.0)
            # self.negative_centroid = nn.Parameter(torch.ones((args.num_classes, dim_negative_centroid), requires_grad=True).cuda()*0.1)
            # self.negative_centroid
            # self.negative_centroid = self.negative_centroid + 1.0
            self.negative_std = []
            self.negative_centroid = []
            for cl in range(args.num_classes):
                # if self.args.weight_cov == 1.0:
                # if self.args.mil_model=='Dsmil':                    
                self.negative_centroid.append(nn.Parameter(torch.zeros((dim_negative_centroid), requires_grad=True).cuda() + self.init_mean))
                self.negative_std.append(nn.Parameter(torch.zeros((dim_negative_centroid), requires_grad=True).cuda() + self.init_std) )
                # else:                    
                #     self.negative_centroid.append(nn.Parameter(torch.zeros((dim_negative_centroid), requires_grad=True).cuda()))
                #     self.negative_std.append(nn.Parameter(torch.zeros((dim_negative_centroid), requires_grad=True).cuda()) )
            # self.negative_std = nn.Parameter(torch.zeros((args.num_classes, dim_negative_centroid), requires_grad=True).cuda())
            # self.negative_std=self.negative_std.add(1.0)
            # self.negative_std = self.negative_std + 1.0
        elif (args.train_instance == 'interinstance_cosine') or (args.train_instance=='intrainstance_cosine'):
            self.negative_centroid = nn.Parameter(torch.zeros((args.num_classes, dim_negative_centroid), requires_grad=True).cuda())

        if hasattr(self, 'negative_centroid'):
            if self.args.optimizer_nc == 'adam':
                self.optimizer['negative_centroid'] = optim.Adam(params=[self.negative_centroid], lr=self.args.lr_center, weight_decay=0.0)
            elif self.args.optimizer_nc == 'adamw':
                self.optimizer['negative_centroid'] = []
                for cl in range(args.num_classes):
                    self.optimizer['negative_centroid'].append(optim.AdamW(params=[self.negative_centroid[cl]]+[self.negative_std[cl]], lr=self.args.lr_center, weight_decay=0.0))
            elif self.args.optimizer_nc == 'sgd':
                self.optimizer['negative_centroid'] = optim.SGD(params=[self.negative_centroid], lr=self.args.lr_center, weight_decay=0.0)

        # args.num_step = len(loader_train)
        if self.args.scheduler_centroid == 'single':
            self.scheduler['negative_centroid'] = []
            for cl in range(args.num_classes):
                self.scheduler['negative_centroid'].append(CosineAnnealingWarmUpSingle(self.optimizer['negative_centroid'][cl], max_lr=self.args.lr_center, epochs=self.args.epochs, steps_per_epoch=self.args.num_step_neg[cl]))
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
    def infer(self, x: torch.Tensor, y):
        """
        <INPUT>
        x: #bags x #instances x #dims => encoded patches

        <OUTPUT> <- None if unnecessary
        Dict
        prob_bag: #bags x #class
        prob_instance: #bags x #instances x #class
        """
        logit_dict = self.forward(x)
        
        if self.args.num_classes == 1:
            if y==0:
                self.std_neg.append(torch.mean(torch.std(logit_dict['feat'], dim=0)).item())
            elif y==1:            
                self.std_pos.append(torch.mean(torch.std(logit_dict['feat'], dim=0)).item())

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
        loss_bag = self.criterion_bag(logit_dict['bag'], Y)
        # print(f'sup loss: {loss_bag.item()}')
        # if 'instance' in logit_dict.keys():
        if self.args.train_instance == 'None':
            return loss_bag
        else:
            loss_instance = getattr(self, self.args.train_instance)(logit_dict['instance'], int(Y[0, 0]))
            return loss_bag + loss_instance
            

    def update(self, X, Y):
        """
        X: #bags x #instances x #dims => encoded patches
        Y: #bags x #classes  ==========> slide-level label

        <OUTPUT>
        None
        """
        # for _optim in self.optimizer.values():
        #     _optim.zero_grad()
            
        for k in self.optimizer.keys():
            if k == 'negative_centroid':
                for _optim in self.optimizer[k]:
                    _optim.zero_grad()
            else:
                self.optimizer[k].zero_grad()

        loss = self.calculate_objective(X, Y)
        loss.backward()
                
        # print(f'negative_centroid lr: {self.optimizer["negative_centroid"][0].param_groups[0]["lr"]}')
        # print(f'mil_model lr: {self.optimizer["mil_model"].param_groups[0]["lr"]}')
        # print(f'[BEFORE] center {Y.item()}: {self.negative_centroid[0][:5]-self.init_mean}')
        # print(f'[BEFORE] std {Y.item()}: {self.negative_std[0][:5]-self.init_std}')
        # for _optim in self.optimizer.values():
        #     _optim.step()
        for k in self.optimizer.keys():
            if k == 'negative_centroid':
                for i in Y[0,:].tolist():
                    i = int(i)
                    if i == 0:
                        self.optimizer[k][i].step()
                        self.scheduler[k][i].step()
            else:
                self.optimizer[k].step()
                if k in self.scheduler.keys():
                    self.scheduler[k].step()

        # print(f'[AFTER] center {Y.item()}: {self.negative_centroid[0][:5]-self.init_mean}')
        # print(f'[AFTER] std {Y.item()}: {self.negative_std[0][:5]-self.init_std}')
        # print(f'============================================================')
        # for k in self.scheduler.keys():            
        #     if k == 'negative_centroid':
        #         self.scheduler[k].step()
    
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
        pseudo_threshold = 0.5
        if target == 0:
            return self.criterion(p, torch.zeros_like(p, device=p.get_device()))
        elif target == 1:
            # pseudo labeling
            with torch.no_grad():
                pseudo_prob = p.sigmoid().unsqueeze(-1)
                pseudo_labeled = (pseudo_prob >= pseudo_threshold).type(torch.LongTensor)
            
            # calculate pseudo labeled loss
            pl_loss = self.criterion(p, pseudo_labeled)
            
            # weighting labeled loss
            pl_loss = self.unlabeled_weight()*pl_loss
                
            return pl_loss

    def semisup2(self, p: torch.Tensor, target=0):
        """ 
        <INPUT>
            p: Length_sequence x Head_num
            target: wsi label. Either 0 or 1

        <return>
            loss: scalar
        """
        pass


    def divdis(self, p: torch.Tensor, target=0):
        """
        p: #instances x ic_dim_out x Head_num
        ic_dim_out == args.num_classes

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
    
    def interinstance_vi(self, p: torch.Tensor, target=0):
# self.negative_centroid = nn.Parameter(torch.zeros((1, dim_negative_centroid), requires_grad=True).cuda())
        if self.args.num_classes == 1:
            return self._interinstance_vi(p, self.negative_centroid, target)
        elif self.args.num_classes > 1:
            loss=0
            # p: Length_sequence x (128 * args.num_classes)
            _p = torch.stack(torch.split(p, [self.ic_dim_out]*self.args.num_classes, dim=1), 2)
            # _p: Length_sequence x 128 x args.num_classes
            for idx in range(self.args.num_classes):
                loss += self._interinstance_vi(_p[:,:,idx], self.negative_centroid[idx:(idx+1), :], 1 if target==idx else 0)
            # _nc = F.normalize(self.negative_centroid, dim=1, eps=1e-8)
            # loss_centroid = (_nc@_nc.T).fill_diagonal_(0)
            # loss += torch.sum(loss_centroid)/(self.args.num_classes*(self.args.num_classes-1))
            return loss
        else:
            raise ValueError('invalid self.args.num_classes')
        
        
    def _interinstance_vi(self, p: torch.Tensor, _negative_centroid: torch.Tensor, target=0):
        """
        1) center(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs
        """

        ls, fs = p.shape
        _p = p - p.mean(dim=0)
        # cross-correlation 해야할것 같지만 시간 없음

        if target == 0:
            dist_negative_centroid = p - _negative_centroid.expand(ls, -1) # _negative_centroid : Length_sequence x fs
            # print(f'var-mean (neg): {torch.mean(dist_negative_centroid)}')
            # print(f'var-max (neg): {torch.amax(torch.abs(dist_negative_centroid))}')
            # print(f'var-min (neg): {torch.amin(torch.abs(dist_negative_centroid))}')
            # print(f'center location: {self.negative_centroid[:,:5]}')
            # return self.args.weight_agree * torch.mean(torch.sqrt(_negative_centroid.var(dim=0) + 0.00000001)) # var loss
            return self.args.weight_agree * dist_negative_centroid.pow_(2).sum().div((ls-1)*fs) # var loss
        elif target == 1:
            dist_negative_centroid = p - _negative_centroid.detach().expand(ls, -1) # _negative_centroid : Length_sequence x fs
            # print(f'var-mean (pos): {torch.mean(dist_negative_centroid)}')
            # print(f'var-max (pos): {torch.amax(torch.abs(dist_negative_centroid))}')
            # print(f'var-min (pos): {torch.amin(torch.abs(dist_negative_centroid))}')
            # return self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree - torch.sqrt(_p.var(dim=0) + 0.00000001))) # variance
            # return -self.args.weight_disagree * torch.mean(dist_negative_centroid.pow_(2)) # variance
            # return self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree - torch.mean(dist_negative_centroid.pow_(2), dim=0))) # variance
            return self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree - dist_negative_centroid.pow_(2).sum(dim=0)/(ls-1))) # variance

        # return loss


    def interinstance_vic(self, p: torch.Tensor, target=0):
# self.negative_centroid = nn.Parameter(torch.zeros((1, dim_negative_centroid), requires_grad=True).cuda())
        if self.args.num_classes == 1:
            return self._interinstance_vic(p, self.negative_centroid[0], self.negative_std[0], target)
        elif self.args.num_classes > 1:
            loss=0
            # p: Length_sequence x (128 * args.num_classes)
            _p = torch.stack(torch.split(p, [self.ic_dim_out]*self.args.num_classes, dim=1), 2)
            # _p: Length_sequence x 128 x args.num_classes
            for cl in range(self.args.num_classes):
                loss += self._interinstance_vic(_p[:,:,cl], self.negative_centroid[cl], self.negative_std[cl], 1 if target==cl else 0)
            # _nc = F.normalize(self.negative_centroid, dim=1, eps=1e-8)
            # loss_centroid = (_nc@_nc.T).fill_diagonal_(0)
            # loss += torch.sum(loss_centroid)/(self.args.num_classes*(self.args.num_classes-1))
            return loss
        else:
            raise ValueError('invalid self.args.num_classes')
        
        
    def _interinstance_vic(self, p: torch.Tensor, _negative_centroid: torch.Tensor, _negative_std: torch.Tensor, target=0):
        """
        1) center(negative)/variance(positive) + 2) Covariance
 
        p: Length_sequence x fs
        """

        ls, fs = p.shape
        self.std_neg
        # _p = p - p.mean(dim=0)
        # print(f'target: {target}')
        if target == 0:
            
            dist_negative_centroid = p - _negative_centroid.unsqueeze(0).expand(ls, -1) # _negative_centroid : Length_sequence x fs
            std = torch.sqrt(dist_negative_centroid.pow(2).sum(dim=0).div(ls-1))
            # dist_negative_centroid_norm = dist_negative_centroid / std
            # corr = (dist_negative_centroid_norm.T @ dist_negative_centroid_norm) / (ls-1)
            # print(f'stddev-mean (neg): {torch.mean(std)}')
            # print(f'stddev-max (neg): {torch.amax((std))}')
            # print(f'stddev-min (neg): {torch.amin((std))}')
            # print(f'center location -neg: {self.negative_centroid[:5]}')
            # return self.args.weight_agree * torch.mean(torch.sqrt(_negative_centroid.var(dim=0) + 0.00000001)) # var loss
            
            # neg = dist_negative_centroid.pow(2).sum().div((ls-1)*fs)
            
            # cov = self.off_diagonal(corr).pow(2).mean()
            # print(f'corr: {cov}')
            # print(f'neg: {neg}')
            # print(f'neg std: {_negative_std}')
            return (self.args.weight_agree * torch.mean(std)) + torch.exp(_negative_std-std.detach().clone()).mean() # var loss
        elif target == 1:
            # __negative_centroid = _negative_centroid.detach().clone()
            dist_negative_centroid = p - _negative_centroid.detach().expand(ls, -1) # _negative_centroid : Length_sequence x fs
            std = torch.sqrt(dist_negative_centroid.pow(2).sum(dim=0).div(ls-1))
            # print(f'stddev-mean (pos): {torch.mean(std)}')
            # print(f'stddev-max (pos): {torch.amax((std))}')
            # print(f'stddev-min (pos): {torch.amin((std))}')
            # print(f'center location -pos: {self.negative_centroid[:5]}')
            # return self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree - torch.sqrt(_p.var(dim=0) + 0.00000001))) # variance
            # return -self.args.weight_disagree * torch.mean(dist_negative_centroid.pow_(2)) # variance
            # return self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree - torch.mean(dist_negative_centroid.pow_(2), dim=0))) # variance
            # pos = torch.mean(F.relu(self.args.stddev_disagree - dist_negative_centroid.pow_(2).sum(dim=0)/(ls-1)))
            
                                                                                                #   torch.sqrt(dist_negative_centroid.pow(2).sum().div((ls-1)*fs))
            # print(f'pos: {pos}')
            return self.args.weight_disagree * torch.mean(F.relu(self.args.stddev_disagree*_negative_std.squeeze(0).detach() - std)) # variance

        # return loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




