import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


# 让多张显卡合作存储和计算 Embedding 层的权重
# 解决单卡显存可能放不下超大词表的问题
class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()

        # 获取当前进程的 rank (第几张卡) 和总 world_size (总卡数)
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # 确保词表总大小能被卡数整除
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings

        # 计算每张卡负责多少个 token 的 embedding
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        # 计算当前卡负责的词表 ID 范围 [start, end)
        # 例如 vocab=100, tp=2:
        # rank 0 负责 [0, 50), rank 1 负责 [50, 100)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 初始化当前卡局部的权重矩阵，行数仅为总行数的 1/tp_size
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        # 绑定权重加载函数，用于从完整权重中切分加载
        self.weight.weight_loader = self.weight_loader

    # 根据当前 GPU 的编号（tp_rank），从完整的预训练权重矩阵中精确地切出属于自己的一部分
    # 并加载进去
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 1. 获取当前卡上参数的 data 句柄
        param_data = param.data
        # 2. 获取切片大小（即当前卡负责多少行）
        # shard_size = vocab_size / tp_size
        shard_size = param_data.size(0)
        # 3. 计算切片的起始位置
        # 例如：shard_size=100
        # rank 0: start_idx = 0 * 100 = 0
        # rank 1: start_idx = 1 * 100 = 100
        start_idx = self.tp_rank * shard_size
        # 4. 核心步骤：从完整的 loaded_weight 中切出属于当前卡的那一部分
        # narrow(dimension, start, length) 表示在第 0 维，从 start_idx 开始取 shard_size 行
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 5. 将切好的权重复制给当前模型的参数
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    # 将隐藏层特征映射到词表大小，得到 logits
    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # context.cu_seqlens_q[1:] - 1 得到每个句子最后一个 token 的位置
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # logits = x @ weight.T
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
