import torch
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # x: [B,in_dim]
        # weight: [in_dim,out_dim]
        # output: [B,out_dim]
        return x @ self.weight + self.bias


# 构造数据：y = 2*x1 + 3*x2 + 1
x = torch.randn(1000, 2)
y = 2 * x[:, 0:1] + 3 * x[:, 1:2] + 1

model = LinearRegression(in_dim=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    # 1. 前向计算
    pred = model(x)

    # 2. 均方误差
    loss = ((pred - y) ** 2).mean()

    # 3. 清空上一轮梯度
    optimizer.zero_grad()

    # 4. 反向传播
    loss.backward()

    # 5. 更新参数
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")

print("weight:", model.weight.detach())
print("bias:", model.bias.detach())