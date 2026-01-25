import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    def __init__(self, model, **kwargs):
        # 动态配置加载
        # 获取 Config 类中定义的所有字段名
        config_fields = {field.name for field in fields(Config)}
        # 从传入的 kwargs 中筛选出属于 Config 的参数，过滤掉无关的参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 实例化配置对象，这样以后增加配置项时不需要修改这里的代码
        config = Config(model, **config_kwargs)

        # 用于存储子进程（Worker）对象的列表
        self.ps = []
        # 用于存储同步事件（Events）的列表，用于主进程唤醒子进程
        self.events = []
        # 强制使用 spawn 启动方式
        # 在涉及 CUDA 的程序中不能使用默认的 fork，否则会引发上下文错误或死锁
        ctx = mp.get_context("spawn")

        # 每次循环启动一个 Worker 进程
        # 主进程自己作为 rank 0
        for i in range(1, config.tensor_parallel_size):
            # 创建一个 Event 对象，用于主进程控制该子进程的同步
            event = ctx.Event()

            # 创建子进程：
            # target=ModelRunner：子进程启动后会直接实例化 ModelRunner
            # args=(config, i, event): 传入配置、当前 Rank ID (i)、以及同步事件
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 初始化主进程：Rank 0
        # 主进程自己也实例化一个 ModelRunner，Rank ID 为 0
        # 注意：主进程持有 self.events（所有子进程的开关列表），说明它是指挥官
        self.model_runner = ModelRunner(config, 0, self.events)
        # 加载 Huggingface 的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 初始化调度器
        # 调度器负责管理 KV Cache 和请求队列，它只在 CPU 上运行，不需要多份
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        # 回收子进程
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            # 如果输入是文本，就使用 __init__ 中加载的 tokenizer 将其转换为 Token ID 列表
            prompt = self.tokenizer.encode(prompt)
        # 封装请求为 Sequence 对象
        seq = Sequence(prompt, sampling_params)
        # 提交给调度器
        self.scheduler.add(seq)

    def step(self):
        # 调度器返回的结果，其中的 seq 表示本轮参与计算的 Sequence 对象列表
        # is_prefill 表示目前所处的状态，bool 类型变量
        seqs, is_prefill = self.scheduler.schedule()
        # 将任务派发给 model_runner 去执行
        # 生成的结果是 token_ids，表示本轮计算中为每个 Sequence 生成的新的 Token ID（Logits 采样之后的结果）
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理
        self.scheduler.postprocess(seqs, token_ids)
        # 收集结果，如果状态是 finished，就返回结果
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        # 如果是 prefill 阶段，就计算本次计算了多少个 Token
        # 如果是 decode 阶段，就计算本次新生成了多少个 token
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # 最接近用户的高级接口
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 是否启用 tqdm 进度条
        if use_tqdm:
            pbar = tqdm(
                total=len(prompts),
                desc="Generating",
                dynamic_ncols=True,
                mininterval=2.0,
            )
        # 参数对齐
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 批量入队
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        # 只要引擎里面还有任务没有 finish，就一直循环执行
        while not self.is_finished():
            t = perf_counter()
            # 执行一步推理
            # output: 本轮刚刚完成的序列列表
            # num_tokens: 正数代表 Prefill 处理的 Token 数，负数代表 Decode 生成的 Token 数
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 结果整理与重排序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 最终解码：将生成的 Token id 转换为人类可以读的自然语言
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
