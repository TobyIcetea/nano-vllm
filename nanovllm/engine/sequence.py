from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    # property 将这个方法转换为只读属性，避免直接修改属性值
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 完成 token 数 = 总 token 数 - 提示 token 数
    # 也就是计算 AI 生成了多少 token
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    # 通过向上取整除法，计算出当前所占用的总块数
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    # 计算最后一个块用了多少个 token，也就是里面填充了多少数据
    # 在下一次生成新 token 的时候，如果 last_block_num_tokens 等于 block_size，
    # 说明当前块已经满了，需要分配新的块
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 获取第 i 个逻辑块中的所有 token
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    # 模型生成了一个新的 token，将这个 token 加到序列中
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 在 Python 中，__getstate__ 和 __setstate__ 用于控制对象如何别 Pickle（序列化）
    # 在多进程或者分布式环境中传输对象的时候，Python 会自动调用它们
    # 这里重写是为了节省带宽，只传输有必要传输的数据
    # 最后的 if else 表示，当处于 prefill 阶段时（num_completion_tokens 为 0），就需要传输所有 token_ids
    # 否则，只需要传输最后一个 token_id，因为前面的 token_id 已经被缓存了
    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    # 如果是 prefill 阶段，就恢复完整列表
    # 如果是 decode 阶段，就只恢复最后一个 token_id
    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
        ) = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
