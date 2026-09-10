import torch
from torch import nn


# LayerNorm
# 1. 求均值
# 2. 求方差
# 3. 设置eps，初始化缩放系数gamma，初始化平移系数beta
# 4. 归一化

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.D = hidden_size
        self.eps = eps

        # gamma: 可学习的缩放参数
        # 初始值为1，不改变归一化的结果
        self.weight = nn.Parameter(torch.ones(hidden_size))

        # beta: 可学习的平移参数
        # 初始值为0
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        # 归一化时，使用fp32精度更高
        input_dtype = x.dtype
        xx = x.float()

        # 对于最后一个维度，求均值【每个token对应所有嵌入向量的维度】
        mean = xx.mean(dim=-1, keepdim=True)

        # 计算方差
        variance = (xx - mean).pow(2).mean(dim=-1, keepdim=True)

        # 归一化
        normalized = (xx - mean) / torch.sqrt(variance + self.eps)

        # 归一化之后替换为原来的精度类型
        normalized.to(input_dtype)

        # 可缩放或可平移
        output = self.weight * normalized + self.bias

        return output




