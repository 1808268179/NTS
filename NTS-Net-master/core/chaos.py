import torch
import torch.nn as nn
import numpy as np

class MultiGranularitySpaceChaos(nn.Module):
    def __init__(self, granularities=[4]):
        super(MultiGranularitySpaceChaos, self).__init__()
        self.granularities = granularities

    def forward(self, x, training=True):
        # 在非训练模式（推理/验证）下，直接返回原始图像，不做任何处理
        if not training:
            return x

        # 破坏逻辑只在训练时执行
        # (为了简化，我们只使用第一个粒度进行破坏)
        G = self.granularities[0]
        batch_size, channels, height, width = x.size()
        
        # 确保输入尺寸可以被G整除
        new_h, new_w = height, width
        if height % G != 0 or width % G != 0:
            new_h = ((height + G - 1) // G) * G
            new_w = ((width + G - 1) // G) * G
            x_padded = nn.functional.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        else:
            x_padded = x

        block_h = new_h // G
        block_w = new_w // G

        # 将图像分割成G*G个块
        blocks = x_padded.view(
            batch_size, channels, G, block_h, G, block_w
        ).permute(
            0, 2, 4, 1, 3, 5
        )  # [B, G, G, C, block_h, block_w]

        shuffled_blocks = torch.zeros_like(blocks)

        for b in range(batch_size):
            source_positions = [(i, j) for i in range(G) for j in range(G)]
            target_positions = source_positions.copy()
            np.random.shuffle(target_positions)

            for (src_i, src_j), (tgt_i, tgt_j) in zip(
                source_positions, target_positions
            ):
                shuffled_blocks[b, tgt_i, tgt_j] = blocks[b, src_i, src_j].clone()

        # 将打乱的块重组成图像
        shuffled_x = (
            shuffled_blocks.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(batch_size, channels, new_h, new_w)
        )
        
        # 如果有填充，裁剪回原始尺寸
        if new_h != height or new_w != width:
            shuffled_x = shuffled_x[:, :, :height, :width]

        return shuffled_x