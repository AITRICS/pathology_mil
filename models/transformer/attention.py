import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Any

from models.transformer.module import Linear


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int, scale: bool = True, attn_drop: float = 0., proj_drop: float = 0., block_mask: list = None) -> None:
        super(ScaledDotProductAttention, self).__init__()
        if scale:
            self.sqrt_dim = np.sqrt(dim)
        else:
            self.sqrt_dim = 1
        self.block_mask = block_mask

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Any] = None
    ) -> Tuple[Tensor, Tensor]:
        # score = ((torch.nan_to_num(torch.bmm(query, key.transpose(1, 2)), nan=0.0, posinf=1.0)) / self.sqrt_dim)  # + 2e-10
        score = ((torch.bmm(query, key.transpose(1, 2))) / self.sqrt_dim)  # + 2e-10
        
        # attn = torch.nan_to_num(F.softmax(score, -1), nan=0.0, posinf=1.0)
        attn = F.softmax(score, -1)
        
        _attn = self.attn_drop(attn)
        context = torch.bmm(_attn, value)
        context = self.proj(context)
        context = self.proj_drop(context)
        
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8, block_mask: list = None) -> None:
        super(MultiHeadAttention, self).__init__()

        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(dim, self.d_head * num_heads)
        self.key_proj = Linear(dim, self.d_head * num_heads)
        self.value_proj = Linear(dim, self.d_head * num_heads)
        
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, scale=True)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        
        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

        return context, attn

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8, block_mask: list = None, sr_ratio: int = 1) -> None:
        super(EfficientMultiHeadAttention, self).__init__()

        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(dim, self.d_head * num_heads)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv1d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            
        self.kv = nn.Linear(dim, dim * 2)
        
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, scale=True)

    def forward(self, x: Tensor, x_kv: Tensor, mask: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0)

        query = self.query_proj(x).view(batch_size, -1, self.num_heads, self.d_head)
        
        if self.sr_ratio > 1:
            x_kv = self.sr(x_kv.permute(0,2,1)).permute(0, 2, 1)
            x_kv = self.norm(x_kv)
        kv = self.kv(x_kv)
        
        kv = kv.reshape(batch_size, -1, 2, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        
        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

        return context, attn