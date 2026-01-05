import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    # 模型名称或路径
    model: str

    # 所有序列的最大总token数
    max_num_batched_tokens: int = 16384

    # 最大并发 sequence 数
    max_num_seqs: int = 512

    # 模型的上下文窗口限制
    max_model_len: int = 4096

    # 显存占用率
    gpu_memory_utilization: float = 0.9

    # 张量并行度，表示使用了几张计算卡
    tensor_parallel_size: int = 1

    # 是否强制使用 eager 模式(但是现在几乎都从来不使用 eager mode)
    enforce_eager: bool = False

    # 从 huggingface 加载的原始配置对象，也就是 config.json 中的内容
    hf_config: AutoConfig | None = None

    # end of sentence，表示结束符的 token 的 id
    # 模型输出遇到这个符号，就会停止输出
    # 当前设置为 -1，之后需要从 config.json 中加载
    eos: int = -1

    # kV Cache 使用的块的大小
    kvcache_block_size: int = 256

    # KV Cache 使用的块的数量
    # 在程序启动之后，会根据可用显存的大小和块的数量，自动计算出有多少可以使用的 kv cache 块
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 防御性编程：用户设置的 max_model_len 不能超过模型本身训练时的极限（max_position_embeddings）
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
