import torch
from torch import nn
import math

class SelfAttention(nn.Module):

    def __int__(self, hidden_states):
        super().__init__()

        self.D = hidden_states

        self.q_proj = nn.Linear(self.D, self.D)
        self.k_proj = nn.Linear(self.D, self.D)
        self.v_proj = nn.Linear(self.D, self.D)

        self.o_proj = nn.Linear(self.D, self.D)

    def forward(self, x):
        B,S,D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1)

        scores = scores / math.sqrt(self.D)

        attention_probs = torch.softmax(scores, dim=-1)

        context = attention_probs @ v

        output = self.o_proj(context)

        return output, attention_probs




