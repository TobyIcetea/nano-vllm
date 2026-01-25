from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        # 存放刚到达，还没有开始处理的请求，通常处于 prefill 阶段
        self.waiting: deque[Sequence] = deque()
        # 存放正在处理、逐个生成 token 的请求，这些请求通常处于 decode 阶段
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # 这段代码采用了一种策略
    # 只要有新任务能跑（Prefill），就优先跑新任务，不去运行 decode
    # 并且当前 batch 中所有的句子只能有一种状态，prefill 或者 decode
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # token数量限制 与 显存限制
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            # 进行资源分配
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            # 取出一个正在运行的任务
            seq = self.running.popleft()
            # 当显存不足以继续 append 新 token 的时候，通过牺牲（抢占）其他任务来腾出空间，保住当前任务
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 牺牲别人（队列尾部的任务，也就是优先级最低、最后进入的那个任务）
                    self.preempt(self.running.pop())
                else:
                    # 牺牲自己（当前任务）
                    self.preempt(seq)
                    break
            else:
                # 进入 else 表示一切正常，分配一个 token 的槽位
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 将本轮调度运行过的任务，按照原有的顺序，重新放回队列的最前端
        # 这种情况主要会出现在“当前运行的任务数量，超过了系统允许的最大并发数”的时候
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            # 更新序列数据
            seq.append_token(token_id)
            # 检查终止条件
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                # 标记结束并释放资源，移出运行队列
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
