# 多头自注意力机制
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super().__init__()
        self.D = d_model
        self.H = num_heads
        self.Dh = d_model // num_heads

        # 权重矩阵qkvo
        self.q_proj = nn.Linear(self.D, self.D, bias=False)
        self.k_proj = nn.Linear(self.D, self.D, bias=False)
        self.v_proj = nn.Linear(self.D, self.D, bias=False)
        self.o_proj = nn.Linear(self.D, self.D, bias=False)

    def forward(self, x, mask=None):
        # batch_size seq_len d_model
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B,T,D] -> [B,T,H,Dh] -> [B,H,T,Dh]
        q = q.view(B, T, self.H, self.Dh).transpose(1, 2)
        k = k.view(B, T, self.H, self.Dh).transpose(1, 2)
        v = v.view(B, T, self.H, self.Dh).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.Dh)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # [B,H,T,T]
        attention_probs = torch.softmax(scores, dim=-1)

        # [B,H,T,T] @ [B,H,T,Dh]
        # -> [B,H,T,Dh]
        context = attention_probs @ v

        # [B,H,T,Dh] -> [B,T,H,Dh] -> [B,T,D]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(B, T, D)
        )

        # [B,T,D] @ [D,D] -> [B,T,D]
        output = self.o_proj(context)

        return output, attention_probs
