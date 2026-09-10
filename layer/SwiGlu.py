from torch import nn

import torch.nn.functional as F


class SwiGlu(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * F.silu(self.gate_proj(x)))