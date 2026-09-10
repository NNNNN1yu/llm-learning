import torch
from torch import nn

class RMSNorm(nn.Module):
    def __int__(self, hidden_size: int, eps: float=1e-6):
        super().__init__()

        self.eps = eps

        # 权重矩阵，全1初始化
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        input_dtype = x.dtype

        xx = x.float()

        squared = xx.pow(2)

        # 对最后一个维度求平方均值
        mean_square = squared.mean(dim=-1, keepdim=True)

        # 计算均方根的倒数
        inverse_rms = torch.rsqrt(mean_square + self.eps)

        # 最终公式
        normalized = inverse_rms * xx

        normalized = normalized.to(input_dtype)

        # 乘可以缩放的权重
        output = normalized * self.weight

        return output








