import math

import torch


def flash_attention(
    q,
    k,
    v,
    q_block_size=64,
    kv_block_size=64,
    is_causal=True
):
    """
    教学版 FlashAttention 前向过程。

    参数：
        q: [B,H,T,Dh]
        k: [B,H,T,Dh]
        v: [B,H,T,Dh]

        q_block_size:
            每个Q块包含多少个token

        kv_block_size:
            每个K/V块包含多少个token

        is_causal:
            是否使用因果Mask

    返回：
        output: [B,H,T,Dh]

    注意：
        这是纯PyTorch教学实现。
        真正的FlashAttention需要使用Triton/CUDA Kernel，
        并手写或优化反向传播。
    """

    if q.dim() != 4:
        raise ValueError(
            "q、k、v必须是[B,H,T,Dh]格式"
        )

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "当前教学实现要求q、k、v形状相同"
        )

    B, H, T, Dh = q.shape

    # Attention缩放系数。
    scale = 1.0 / math.sqrt(Dh)

    # 分块计算后，将每一个Q块的结果放入这里。
    output_blocks = []

    # ==================================================
    # 外层循环：遍历Q块
    # ==================================================
    for q_start in range(0, T, q_block_size):
        q_end = min(
            q_start + q_block_size,
            T
        )

        # 当前Q块：
        #
        # [B,H,T,Dh]
        #       ↓
        # [B,H,Bq,Dh]
        #
        # Bq表示当前Q块的长度。
        q_block = q[:, :, q_start:q_end, :]

        Bq = q_end - q_start

        # m表示当前已经处理过的Key中，
        # 每个Query对应的最大attention score。
        #
        # shape: [B,H,Bq,1]
        #
        # 初始时还没有处理任何Key，
        # 因此最大值设为负无穷。
        running_max = torch.full(
            (B, H, Bq, 1),
            float("-inf"),
            device=q.device,
            dtype=torch.float32
        )

        # l表示Softmax分母的累计值：
        #
        # sum(exp(score - max_score))
        #
        # shape: [B,H,Bq,1]
        running_sum = torch.zeros(
            (B, H, Bq, 1),
            device=q.device,
            dtype=torch.float32
        )

        # accumulator表示尚未除以Softmax分母的
        # 加权Value累计结果。
        #
        # shape: [B,H,Bq,Dh]
        accumulator = torch.zeros(
            (B, H, Bq, Dh),
            device=q.device,
            dtype=torch.float32
        )

        # ==================================================
        # 内层循环：遍历K/V块
        # ==================================================
        for kv_start in range(0, T, kv_block_size):
            kv_end = min(
                kv_start + kv_block_size,
                T
            )

            # 因果注意力中，如果整个K/V块都位于
            # 当前Q块所有位置的未来，就不需要计算。
            #
            # 当前Q块最大位置为q_end-1。
            # 如果kv_start >= q_end，
            # 那么这个K/V块全部属于未来位置。
            if is_causal and kv_start >= q_end:
                break

            # 当前K/V块：
            #
            # [B,H,T,Dh]
            #       ↓
            # [B,H,Bk,Dh]
            k_block = k[:, :, kv_start:kv_end, :]
            v_block = v[:, :, kv_start:kv_end, :]

            # 1. 只计算当前Q块与当前K块的分数。
            #
            # q_block:                   [B,H,Bq,Dh]
            # k_block.transpose:         [B,H,Dh,Bk]
            # block_scores:              [B,H,Bq,Bk]
            #
            # 不会生成完整的[B,H,T,T]矩阵。
            block_scores = (
                q_block
                @ k_block.transpose(-2, -1)
            ) * scale

            # 使用FP32完成Softmax相关计算，
            # 避免FP16/BF16指数运算不稳定。
            block_scores = block_scores.float()

            # 2. 添加因果Mask。
            if is_causal:
                # 当前Q块的绝对位置。
                #
                # shape: [Bq]
                q_positions = torch.arange(
                    q_start,
                    q_end,
                    device=q.device
                )

                # 当前K块的绝对位置。
                #
                # shape: [Bk]
                k_positions = torch.arange(
                    kv_start,
                    kv_end,
                    device=q.device
                )

                # Query位置只能关注：
                #
                # key_position <= query_position
                #
                # [Bq,1] 与 [1,Bk] 广播得到：
                # [Bq,Bk]
                causal_mask = (
                    k_positions.unsqueeze(0)
                    <= q_positions.unsqueeze(1)
                )

                # [Bq,Bk]
                #      ↓ 增加batch和head维度
                # [1,1,Bq,Bk]
                causal_mask = (
                    causal_mask
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                block_scores = (
                    block_scores.masked_fill(
                        ~causal_mask,
                        float("-inf")
                    )
                )

            # 3. 找到当前分块的最大分数。
            #
            # [B,H,Bq,Bk]
            #       ↓
            # [B,H,Bq,1]
            block_max = block_scores.amax(
                dim=-1,
                keepdim=True
            )

            # 4. 计算包含历史分块和当前分块的
            # 新全局最大值。
            #
            # shape: [B,H,Bq,1]
            new_running_max = torch.maximum(
                running_max,
                block_max
            )

            # 5. 当最大值发生变化时，
            # 之前累计的结果必须重新缩放。
            #
            # old_scale:
            # exp(old_max - new_max)
            #
            # shape: [B,H,Bq,1]
            old_scale = torch.exp(
                running_max - new_running_max
            )

            # 6. 计算当前块的未归一化Softmax概率。
            #
            # exp(score - new_max)
            #
            # shape: [B,H,Bq,Bk]
            block_exp = torch.exp(
                block_scores - new_running_max
            )

            # 7. 更新Softmax分母。
            #
            # 旧分母需要按照新的最大值重新缩放。
            #
            # new_sum =
            # old_scale * old_sum
            # + sum(exp(current_scores - new_max))
            #
            # shape: [B,H,Bq,1]
            new_running_sum = (
                old_scale * running_sum
                + block_exp.sum(
                    dim=-1,
                    keepdim=True
                )
            )

            # 8. 更新Value加权和。
            #
            # block_exp: [B,H,Bq,Bk]
            # v_block:   [B,H,Bk,Dh]
            #
            # block_exp @ v_block:
            #             [B,H,Bq,Dh]
            #
            # 旧累计值同样需要重新缩放。
            accumulator = (
                old_scale * accumulator
                + block_exp @ v_block.float()
            )

            # 保存新的最大值和分母，
            # 供下一个K/V块继续使用。
            running_max = new_running_max
            running_sum = new_running_sum

        # 9. 所有K/V块处理完成后，
        # 除以完整Softmax分母。
        #
        # accumulator: [B,H,Bq,Dh]
        # running_sum: [B,H,Bq,1]
        #
        # output_block: [B,H,Bq,Dh]
        output_block = (
            accumulator
            / running_sum.clamp_min(1e-12)
        )

        # 恢复原始输入数据类型。
        output_block = output_block.to(q.dtype)

        output_blocks.append(output_block)

    # 将所有Q块沿序列维拼接：
    #
    # 多个[B,H,Bq,Dh]
    #       ↓
    # [B,H,T,Dh]
    output = torch.cat(
        output_blocks,
        dim=2
    )

    return output