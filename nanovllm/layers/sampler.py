import torch
from torch import nn


# 为 batch 中的每个句子，根据 logits 和 temperature 采样一个 token
class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float() / temperatures.unsqueeze(-1)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled
