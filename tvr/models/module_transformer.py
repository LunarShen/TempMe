from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from timm.models.layers import drop_path
import torch
from torch import nn
import math
import torch.nn.functional as F
from .until_module import LayerNorm, ACT2FN
from collections import OrderedDict

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, lora_dim): 
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5
        self.lora_dim = lora_dim

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.TVPt_LoRA_a = nn.Parameter(torch.zeros(lora_dim, embed_dim))
        nn.init.kaiming_uniform_(self.TVPt_LoRA_a, a=math.sqrt(5))
        self.TVPt_LoRA_b = nn.Parameter(torch.zeros(3 * embed_dim, lora_dim))
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, x, attn_mask):
        bsz, tgt_len, embed_dim = x.size()
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    
        qkv_delta = F.linear(x, self.TVPt_LoRA_a)
        qkv_delta = F.linear(qkv_delta, self.TVPt_LoRA_b).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        q,k,v = q+q_delta,k+k_delta,v+v_delta
        
        q = q * self.scaling
        attn = (q @ k.transpose(-2, -1))
        
        attn += attn_mask[:,None,:,:]
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
        
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_dim: int, attn_mask=None):
        super(ResidualAttentionBlock, self).__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = Attention(d_model, n_head, lora_dim)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
        
    def attention(self, x: torch.Tensor, attn_mask_: torch.Tensor):

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device)
        return self.attn(x, attn_mask_)

    def forward(self, para_tuple: tuple):
        x, attn_mask = para_tuple
                          
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, lora_dim: int, attn_mask=None):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, lora_dim) for _ in range(layers)])
        
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]