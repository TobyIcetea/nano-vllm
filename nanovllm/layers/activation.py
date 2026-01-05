import torch
from torch import nn
import torch.nn.functional as F


# 对输入张量进行 SiLU 激活函数和乘法操作
class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
