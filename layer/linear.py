import torch
from torch import nn


class LinearLayer(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        self.bias = nn.Parameter(torch.randn(out_size)) if bias else None

    def forward(self, x):
        output = x @ self.weight.t()
        if self.bias is not None:
            output += self.bias
        return output