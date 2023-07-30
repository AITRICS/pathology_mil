import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .mil import MilBase
    
class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1) # 1 x C
        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier, instance_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.instance_classifier = instance_classifier
    def forward(self, x):
        feats, classes = self.i_classifier(x) # feats = instance embedding [seq x dim_in]
        prediction_bag, A, B = self.b_classifier(feats, classes) # A =  instance_logit [seq x 1], B = bag embedding [1,1,dim_in]
        
        logit_instances = self.instance_classifier(feats)
        return classes, prediction_bag, A, B , logit_instances
        
        # args=args, optimizer=None, criterion=None, scheduler=None, dim_in=dim_in, dim_latent=512, dim_out=args.num_classes
class Dsmil(MilBase):
    def __init__(self, encoder=None, **kwargs):
        super().__init__(**kwargs)
        
        self.i_classifier = FCLayer(in_size=self.dim_in, out_size=self.dim_out)
        self.b_classifier = BClassifier(input_size=self.dim_in, output_class=self.dim_out)
        self.milnet = MILNet(self.i_classifier, self.b_classifier, self.instance_classifier)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.5, 0.9), weight_decay=0.005)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.num_epochs, 0.000005)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs*self.args.num_step, 0.000005)
        
    def forward(self, x: torch.Tensor):
        dsmil_input = x.squeeze(0)
        # logit_instance: #instance x self.dim_out
        # logit_bag: 1 x self.dim_out
        logit_instance, logit_bag, _, _, logit_instances = self.milnet(dsmil_input) # ins_prediction (num_patch, n) bag_prediction (1,n)        
        return logit_bag, logit_instance.unsqueeze(0) ,logit_instances
        # average 해야함
    
    def calculate_objective(self, X, Y):
        logit_bag, logit_instance, logit_instances = self.forward(X) # bs x num_seq x c
        max_logit_instance, _ = torch.max(logit_instance, 1)        # (1,n)
        bag_loss = self.criterion(logit_bag.view(1, -1), Y.view(1, -1)) # num class n : BCE([1,n],[1,n]), BCEWithLogitsLoss()
        max_loss = self.criterion(max_logit_instance.view(1, -1), Y.view(1, -1))
        
        if self.aux_loss != 'None':
            loss2 = self.criterion_aux(logit_instances, Y[0, 0])
        else:
            loss2 = None
        
        return 0.5*bag_loss + 0.5*max_loss, loss2
    
    def infer(self, x: torch.Tensor):
        logit_bag, logit_instance ,_= self.forward(x)
        max_logit_instance, _ = torch.max(logit_instance, 1)        # (1,n)
        return 0.5*torch.sigmoid(max_logit_instance)+0.5*torch.sigmoid(logit_bag), None