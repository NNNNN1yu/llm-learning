import math

import torch
from torch import nn

class GroupQueryAttention(nn.Module):
    def __init__(self, nums_heads:int, hidden_size: int, max_seq_len: int, nums_groups: int):
        super().__init__()
        assert hidden_size % nums_heads == 0
        assert nums_heads % nums_groups == 0

        self.N = nums_heads
        self.H = hidden_size
        self.S = max_seq_len
        self.G = nums_groups
        self.Nh = hidden_size // nums_heads
        self.Ng = nums_heads // nums_groups

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # rope计算会用到hidden_size和max_seq_len
        # self.rope = RopeEmbedding(self.H, self.S)

    def forward(self, x, mask=None):

        B,S,H = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B,S,self.N,self.Nh).transpose(1, 2)
        # kv的头数量改为self.N / nums_groups = self.Ng  每一维的数量为num_groups
        k = k.view(B,S,self.G,self.Nh).transpose(1, 2)
        v = v.view(B,S,self.G, self.Nh).transpose(1, 2)

        # 针对与kv[B,N,S,Nh] 在第二维复制
        k = k.repeat_interleave(self.Ng, dim = 1)
        v = v.repeat_interleave(self.Ng, dim = 1)

        q = self.rope(q)
        k = self.rope(k)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.Nh)
        if mask is not None:
            scores = scores.mask_fill(~mask, float('-inf'))

        attention_probs = torch.softmax(scores, dim=-1)

        # 转换维度：
        output = attention_probs @ v
        output = output.transpose(1,2).contiguous().view(B,S,H)

        return self.o_proj(output)



