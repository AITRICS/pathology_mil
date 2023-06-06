import torch.nn as nn
from torch import Tensor
from typing import Tuple
from models.transformer.attention import MultiHeadAttention
from models.transformer.module import PositionalEncoding, FeedForward, LayerNorm, FeedForwardUseConv
from models.transformer.utils import *

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,             # dimension of transformer model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.attention_prenorm = LayerNorm(d_model)
        self.feed_forward_prenorm = LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        # self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.feed_forward = FeedForwardUseConv(d_model, d_ff, dropout_p)

    def forward(self, inputs: Tensor, self_attn_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, attn
    
class TransformerEncoderLayer_Afternorm(nn.Module):
    def __init__(
            self,
            d_model: int = 512,             # dimension of transformer model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ) -> None:
        super(TransformerEncoderLayer_Afternorm, self).__init__()
        self.attention_prenorm = LayerNorm(d_model)
        self.feed_forward_prenorm = LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardUseConv(d_model, d_ff, dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs: Tensor, self_attn_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        # INPUT:  minibatch x token x dim
        # OUTPUT: minibatch x token x dim
        residual = inputs
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs = self.dropout(outputs)
        outputs += residual
        outputs = self.attention_prenorm(outputs)

        residual = outputs
        outputs = self.feed_forward(outputs)
        outputs += residual
        outputs = self.feed_forward_prenorm(outputs)

        return outputs, attn

class TransformerEncoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, 
                 d_input: int, 
                 n_layers: int, 
                 n_head: int,
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1, 
                 pe_maxlen: int = 5000, 
                 use_pe: bool = True):
        super(TransformerEncoder, self).__init__()
        # parameters
        self.use_pe = use_pe
        self.input_linear = False

        if d_input != d_model:
            # use linear transformation with layer norm to replace input embedding
            self.linear_in = nn.Linear(d_input, d_model)
            self.input_linear = True

        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer_Afternorm(
                d_model=d_model,
                num_heads=n_head,
                d_ff=d_ff,
                dropout_p=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, enc_input):
        # enc_input:  minibatch x token x dim
        # enc_output: minibatch x token x dim

        enc_output = self.dropout(
            self.layer_norm_in(enc_input))
           
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, None)
        
        return enc_output

