import math

import torch
from torch import nn

class MultiHeadAttentionWithRope(nn.Module):
    def __init__(self, nums_heads:int, hidden_size: int, max_seq_len: int):
        super().__init__()
        assert hidden_size % nums_heads == 0

        self.N = nums_heads
        self.H = hidden_size
        self.S = max_seq_len
        self.Nh = hidden_size // nums_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # rope计算会用到hidden_size和max_seq_len
        # self.rope = RopeEmbedding(self.H, self.S)

    def forward(self, x, mask=None, past_key_values=None):

        B,S,H = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B,S,self.N,self.Nh).transpose(1, 2)
        k = k.view(B,S,self.N,self.Nh).transpose(1, 2)
        v = v.view(B,S,self.N,self.Nh).transpose(1, 2)

        # 计算rope
        past_len = 0
        # past_key_values的形状是 B,H,S_past,Nh 即：每个batch对应的每个头的每个past_token的k,v都记录下来
        if past_key_values is not None:
            _, _, past_len,_ = past_key_values[0].shape()

        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)

        if past_key_values is not None:
            # 再第三维度拼接起来 B,H,S_past,Nh --> B,H,S_past+1,Nh
            past_key, past_value = past_key_values[0]

            k = torch.cat((past_key, q), dim=-2)
            v = torch.cat((past_value, q), dim=-2)

        current_key_value = (k,v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.Nh)
        if mask is not None:
            scores = scores.mask_fill(~mask, float('-inf'))

        attention_probs = torch.softmax(scores, dim=-1)

        # 转换维度：
        output = attention_probs @ v
        output = output.transpose(1,2).contiguous().view(B,S,H)

        return self.o_proj(output), current_key_value, attention_probs



