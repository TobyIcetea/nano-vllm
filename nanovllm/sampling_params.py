from dataclasses import dataclass


# dataclass 装饰器表示给类添加 __init__、__repr__、__eq__ 等方法
@dataclass
class SamplingParams:
    # 温度：控制文本生成的随机性
    temperature: float = 1.0
    # 最大生成 token 数
    max_tokens: int = 64
    # 是否忽略 EOS 结束符标记
    ignore_eos: bool = False

    def __post_init__(self):
        # 不允许贪婪采样（即温度接近 0）
        # 根据 softmax 时候的计算公式，如果温度等于 0，计算中会发生除 0 错误
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
