''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
from transformer.utils import seqlen2cu_len
class MultiHeadSelfAttention_Flash(nn.Module):
    ''' Multi-Head self Attention module with Flash Attention '''

    def __init__(self, n_head, d_model, d_qkv, dropout=0.1, causal=False):
        super().__init__()
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.w_qkv = nn.Linear(d_model, 3 * n_head * d_qkv, bias=False)
        self.w_o = nn.Linear(n_head * d_qkv, d_model, bias=False)
        self.dropout_rate = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal
    def forward(self, x, seq_lens):
        # x should be of shape (total_len, d_model)
        # seq_lens should be of shape (batch_size,) like (4, 3, 2, 23, 1, ...)
        drop_rate = self.dropout_rate if self.training else 0.0
        residual = x
        ################# YOUR CODE HERE #################
        # Hints: 
        # 1. use self.w_qkv to get qkv
        # 2. change seq_lens to cu_seqlens by using seqlen2cu_len function
        # 3. get max_len from seq_lens
        ##################################################
        qkv = self.w_qkv(x)  # (total_tokens, 3 * n_head * d_qkv)
        qkv = qkv.view(-1, 3, self.n_head, self.d_qkv).contiguous()
        # FlashAttention expects packed qkv tensor named qkv_packed
        qkv_packed = qkv
        cu_seqlens = seqlen2cu_len(seq_lens.to(dtype=torch.int32, device=x.device))
        max_len = int(seq_lens.max().item())
        
        output = flash_attn_varlen_qkvpacked_func(qkv_packed, cu_seqlens, max_len, dropout_p=drop_rate, causal=self.causal)
        output = output.reshape(-1, self.n_head * self.d_qkv)
        ################# YOUR CODE HERE ###################
        # Hints:
        # 1. use self.w_o to project output back to d_model
        # 2. apply dropout, residual connection and layer norm
        ##################################################
        output = self.w_o(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(output + residual)
        return output

class MultiHeadCrossAttention_Flash(nn.Module):
    def __init__(self, n_head, d_model, d_qkv, dropout=0.1, causal=False):
        super().__init__()
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.w_q = nn.Linear(d_model, n_head * d_qkv, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * n_head * d_qkv, bias=False)
        self.w_o = nn.Linear(n_head * d_qkv, d_model, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_rate = dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal
    def forward(self, x_q, x_kv, seq_lens_q, seq_lens_kv):
        # x_q should be of shape (total, d_model)
        # x_kv should be of shape (total, d_model)
        # seq_lens_q should be of shape (batch_size,) like (4, 3, 2, 23, 1, ...)
        # seq_lens_kv should be of shape (batch_size,) like (4, 3, 2, 23, 1, ...)
        drop_rate = self.dropout_rate if self.training else 0.0 # flash attention won't change dropout rate to 0 during eval automatically
        residual = x_q
        ################# YOUR CODE HERE #################
        # Hints: 
        # 1. use self.w_q, self.w_kv to get q, k, v
        # 2. change seq_lens to cu_seqlens by using seqlen2cu_len function
        # 3. get max_len_q and max_len_kv from seq_lens_q
        ##################################################
        q = self.w_q(x_q).view(-1, self.n_head, self.d_qkv)
        kv = self.w_kv(x_kv).view(-1, 2, self.n_head, self.d_qkv)
        k = kv[:, 0, ...]
        v = kv[:, 1, ...]
        cu_seqlens_q = seqlen2cu_len(seq_lens_q.to(dtype=torch.int32, device=x_q.device))
        cu_seqlens_kv = seqlen2cu_len(seq_lens_kv.to(dtype=torch.int32, device=x_q.device))
        max_len_q = int(seq_lens_q.max().item())
        max_len_kv = int(seq_lens_kv.max().item())
        # output shape: (total_tokens_q, n_head, d_kv)
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_len_q,
            max_seqlen_k=max_len_kv,
            dropout_p=drop_rate,
            causal=self.causal, # flash attention will handle causal masking inside
        )
        ################# YOUR CODE HERE ###################
        # Hints:
        # 1. use self.w_o to project output back to d_model
        # 2. apply dropout, residual connection and layer norm
        ##################################################
        output = output.reshape(-1, self.n_head * self.d_qkv)
        output = self.w_o(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(output + residual)
        return output
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # SwiGLU: first linear outputs 2*d_hid for gating, effective hidden dim remains d_hid
        self.w_1 = nn.Linear(d_in, 2 * d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x_proj = self.w_1(x)
        x_a, x_b = x_proj.chunk(2, dim=-1)
        x = self.w_2(F.silu(x_a) * x_b)
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
