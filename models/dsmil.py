import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .milbase import MilBase
    
class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(in_size, in_size), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        _feat_instance = self.encoder(feats)
        x = self.fc(_feat_instance)
        return _feat_instance, x

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

        self.input_size = input_size
        self.output_class = output_class

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C, N = # of instances
        num_instance = feats.shape[0]
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

        # A_expand = A.unsqueeze(2).expand(-1, -1, self.input_size) # N C --> N C 1 --> N C V
        # V_expand = V.unsqueeze(1).expand(-1, self.output_class, -1) # N V --> N 1 V --> N C V

        # return C, A, B, (A_expand*V_expand).squeeze(1)
        return C, A, B
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feat_instance, instance_logit_stream1 = self.i_classifier(x)
        # prediction_bag, A, B, feat_instance = self.b_classifier(feats, instance_logit_stream1)
        prediction_bag, A, B = self.b_classifier(feat_instance, instance_logit_stream1)
        
        # return instance_logit_stream1, prediction_bag, A, B, feat_instance
        return instance_logit_stream1, prediction_bag, A, B, feat_instance
        
# args=args, optimizer=None, criterion=None, scheduler=None, dim_in=dim_in, dim_latent=512, dim_out=args.num_classes
class Dsmil(MilBase):
    def __init__(self, args, ma_dim_in):
        super().__init__(args=args, ma_dim_in=ma_dim_in, ic_dim_in=ma_dim_in)
        self.args=args
        self.dim_out=args.num_classes 
        self.i_classifier = FCLayer(in_size=ma_dim_in, out_size=self.dim_out)
        self.b_classifier = BClassifier(input_size=ma_dim_in, output_class=self.dim_out, passing_v=True if args.passing_v==1 else False)
        self.milnet = MILNet(self.i_classifier, self.b_classifier)
        # self.optimizer={}
        # self.scheduler={}

        if args.train_instance == 'None':
            self.optimizer['mil_model'] = torch.optim.Adam(list(self.i_classifier.parameters())+list(self.b_classifier.parameters())+
                                                            list(self.milnet.parameters()), lr=args.lr, betas=(0.5, 0.9), weight_decay=0.005)
        else:
            self.optimizer['mil_model'] = torch.optim.Adam(list(self.i_classifier.parameters())+list(self.b_classifier.parameters())+
                                                            list(self.milnet.parameters())+list(self.instance_classifier.parameters()), lr=args.lr, betas=(0.5, 0.9), weight_decay=0.005)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.num_epochs, 0.000005)
        self.scheduler['mil_model'] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer['mil_model'], self.args.epochs*self.args.num_step, 0.000005)
        
    def forward(self, x: torch.Tensor):
        dsmil_input = x.squeeze(0)
        # logit_instance: #instance x self.dim_out
        # logit_bag: 1 x self.dim_out
        instance_logit_stream1, logit_bag, _, _, feat_instance = self.milnet(dsmil_input) # ins_prediction (num_patch, n) bag_prediction (1,n)
        if self.args.train_instance != 'None':
            logit_instance = self.instance_classifier(feat_instance)
            # logit_bag: 1 x num_class, instance_logit_stream1: #instances x num_class, logit_instance: #instances (x num_class) x ic_dim_out(=V) (x args.ic_num_head)
            # logit_bag: #bags x args.output_bag_dim     logit_instances: #instances x ic_dim_out (x Head_num)
            return {'bag': logit_bag, 'instance_stream1': instance_logit_stream1.unsqueeze(0), 'instance': logit_instance}
        else:
            return {'bag': logit_bag, 'instance_stream1': instance_logit_stream1.unsqueeze(0)}
    
    def calculate_objective(self, X, Y):
        logit_dict = self.forward(X)
        max_logit_instance, _ = torch.max(logit_dict['instance_stream1'], 1)        # (1,n)
        bag_loss = self.criterion_bag(logit_dict['bag'].view(1, -1), Y.view(1, -1)) # num class n : BCE([1,n],[1,n]), BCEWithLogitsLoss()
        max_loss = self.criterion_bag(max_logit_instance.view(1, -1), Y.view(1, -1))
            
        if self.args.train_instance == 'None':
            return 0.5*bag_loss + 0.5*max_loss
        else:
            loss_instance = getattr(self, self.args.train_instance)(logit_dict['instance'], int(Y[0, 0]))
            return 0.5*bag_loss + 0.5*max_loss + loss_instance
    
    def infer(self, x: torch.Tensor):
        logit_dict = self.forward(x)
        max_logit_instance, _ = torch.max(logit_dict['instance_stream1'], 1)        # (1,n)
        return 0.5*torch.sigmoid(max_logit_instance)+0.5*torch.sigmoid(logit_dict['bag']), None