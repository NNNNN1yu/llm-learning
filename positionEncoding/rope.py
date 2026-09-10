import torch
from torch import nn

"""
RoPE位置编码
根据 token 的位置，把 Q、K 的特征每两个维度组成一组，然后在二维平面中旋转。
RoPE 对每个 batch、每个 head、每个 token 分别操作，不改变形状。

输入：多头拆分后的张量 x: [B,H,T,Dh]   B：batch size H：注意力头数 T：序列长度 Dh：每个头的维度

"""

class RopePositionEncoder(nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int, rope_base: float=10000.0):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("Hidden size must be an even number.")

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

        # 计算每个二维特征对的旋转频率
        # theta_i = 1 / base^(2i / head_dim)
        # theta 的 维度数为 hidden_size / 2
        theta = 1 / (self.rope_base ** (torch.arange(0, self.hidden_size, 2, dtype=torch.float32) / self.hidden_size))

        # 生成所有位置编号, pos_ids 的维度数为 max_seq_len
        pos_ids = torch.arange(max_seq_len, dtype=torch.float32)

        # 3. 计算每个位置、每个维度对的旋转角度。
        # pos_ids: [max_seq_len] -- theta: [head_dim/2] --> outer: [max_seq_len, head_dim/2]
        # 得到每个token对应的所有二维特征对的旋转频率，这个时候只计算了旋转角度，并没有特征值
        freq = torch.outer(pos_ids, theta)

        # 4. 计算每个token sin 和 cos --> sin/cos:[max_seq_len, head_dim/2]
        sin, cos = torch.sin(freq), torch.cos(freq)

        # sin/cos 不需要训练，但需要随模型迁移到 GPU， 因此注册为 buffer，而不是 Parameter。
        self.register_buffer("sin_table", sin, persistent=False)
        self.register_buffer("cos_table", cos, persistent=False)

    def forward(self, x, offset: int = 0):
        """
        输入：
            x: [B,H,T,Dh]
            offset: 当前序列的起始位置，生成时用于 KV Cache

        输出：
            rotated_x: [B,H,T,Dh]
        """

        # 转为多头注意力之后的形状 batch_size num_heads seq_len hidden_size
        B, H, seq_len, hidden_size = x.shape

        # 取出当前位置需要的 sin/cos。
        # 训练时 offset 通常为 0： [0, 1, ..., seq_len-1]
        # KV Cache 生成时可能为： [past_len, ..., past_len+seq_len-1]
        # shape: [T,Dh/2]
        sin = self.sin_table[offset:offset + seq_len]
        cos = self.cos_table[offset:offset + seq_len]

        # 避免 FP16/BF16 输入与 FP32 sin/cos 运算后，输出被提升成 FP32。
        sin = sin.to(dtype=x.dtype)
        cos = cos.to(dtype=x.dtype)

        # 把最后一个维度按照奇偶位置拆成两部分。
        # x1 取维度：0,2,4,6,...
        # x2 取维度：1,3,5,7,...
        # [B,H,T,Dh] -> [B,H,T,Dh/2]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # 对每个二维向量 (x1,x2) 进行旋转：
        # x1' = x1*cos - x2*sin
        # x2' = x1*sin + x2*cos
        # sin/cos: [T,Dh/2]
        # x1/x2:   [B,H,T,Dh/2]
        # PyTorch 自动在 B、H 维广播。
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # 将每组旋转后的两个分量重新交错合并。
        # stack:
        # 两个 [B,H,T,Dh/2]
        #     ↓
        # [B,H,T,Dh/2,2]
        rotated_pair = torch.stack((rotated_x1, rotated_x2), dim=-1)

        # flatten(-2) 合并最后两个维度：
        # [B,H,T,Dh/2,2]
        #     ↓
        # [B,H,T,Dh]
        rotated_x = rotated_pair.flatten(-2)

        # 最后没有用cat是因为要按照原有的顺序拼接

        return rotated_x
