import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "hccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.npu.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("npu")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                # 谁申请，谁释放
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.npu.synchronize()
        dist.destroy_process_group()

    def loop(self):
        # 这是一个死循环
        # 只有 rank > 0 的工作进程才会执行
        # 作用是听取 rank == 0 进程的指令
        while True:
            # 这里会卡住（阻塞），直到共享内存里有了新数据，且 Event 信号被触发
            # 解析出来的 method_name 是字符串（比如 "forward"），args 是参数列表
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        # 防御性编程：只有 worker 才去读指令
        assert self.world_size > 1 and self.rank > 0
        # wait
        self.event.wait()
        # 实际上作用类似于一个 queue
        # 不过为了极致的性能，nano-vllm 选择操作字节流
        # 这里相当于是一种约定：读取协议头，小端序，前 4 个字节代表「数据端的长度」
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 读取协议体，并且通过 pickle 将字节数据转换为 python 对象
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        # 返回读取到的「要执行的函数名 + 参数列表」
        return method_name, args

    def write_shm(self, method_name, *args):
        # 防御性编程：只有 rank == 0 进程才去写指令
        assert self.world_size > 1 and self.rank == 0
        # 将数据（函数名 + 参数列表）使用 pickle 序列化为字节流
        data = pickle.dumps([method_name, *args])
        # 写入协议头，小端序，前 4 个字节代表「数据端的长度」
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        # 写入协议体，即数据端
        self.shm.buf[4 : n + 4] = data
        # 触发 Event 信号，通知 worker 进程有新数据可读
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        # rank==0 指挥 worker 工作
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # rank==0 自己工作
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        # 清空当前未使用的显存缓存
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        # max_num_batched_tokens 是一次推理时，最大允许的 token 数
        # max_model_len 是一个序列最长的长度
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        # num_seqs: 按照上面的两个 max 标准，我最多可以处理多少个 seq
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        # 生成假数据并运行
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.npu.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # free 和 total 表示当前显存的空闲量 和 总量
        free, total = torch.npu.mem_get_info()
        used = total - free
        # 在 warmup_model 执行的过程中，显存占用达到的最大值
        peak = torch.npu.memory_stats()["allocated_bytes.all.peak"]
        # 当前占用了多大的显存
        current = torch.npu.memory_stats()["allocated_bytes.all.current"]
        # 计算 KV 头数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 一个 block 会占用多少显存
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )
        # 我们可以申请多少个 block
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        # 一次性预分配：申请一大块儿未初始化的显存
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 寻找使用 block 最多的 seq
        # 每个 seq 都有自己的 block_table 属性，例如 [1, 45, 3]
        # 表示这个 seq 的数据被分散在了 1、45、3 这几个显存 block 中
        # 不同的 seq 长度不同，用到的 block 数量也不一样。这里找出这个 batch 中谁用的块最多
        max_len = max(len(seq.block_table) for seq in seqs)
        # 填充对齐（Padding）
        # 为了将所有的列表合并成一个 GPU Tensor（矩阵），它们必须长度一致
        # 短的 Sequence 后面补 -1
        # 比如 max_len 是 3，此时 Seq A 就是 [10, 20] -> [10, 20, -1]
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        # 将数据搬运到 GPU
        # 其中，pin_memort=True 表示在 CPU 上申请“锁页内存”。这块内存不会被 OS 交换到硬盘 Swap 分区中，允许 GPU 通过 DMA 直接读取，传输速度快很多
        # non_blocking=True 表示异步传输。CPU 发出“搬运”指令之后，不等搬完就往下执行。这能让 CPU 准备在一个数据和 GPU 接收当前数据并行发生
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).npu(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # 扁平化（Flattening）
            # 不再使用 padding 把大家补齐成一样的长度，而是直接首尾相连拼起来
            # Seq A (Hello) + Seq B (World!) -> [Hello, World!]
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            # 增量 prefill
            # 如果 num_cached_tokens > 0，说明系统复用了之前的计算结果
            # 极大地降低了 TTFT 首字延迟
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen

            # 累计长度（Cumulative Lengths - CSR Format）
            # 记录每个 Sequence 在大长条辽宁的分割点
            # 假设 Seq A 长 5，Seq B 长 3
            # cu_seqlens = [0, 5, 8]
            # GPU 看到这个就知道，0~5 是第一个 seq，5~8 是第二个 seq
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                # 算出物理块的起始物理地址
                # 比如 Block ID 10, Block Size 16 -> start = 160
                start = seq.block_table[i] * self.block_size
                # 处理边界：如果是最后一个 Block，可能没填满，只映射用到的部分
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                # 将物理地址加入列表
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).npu(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).npu(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).npu(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).npu(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).npu(non_blocking=True)
        # 上下文传递（创建上下文）
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # 只取最后一个，也就是上一步最后刚刚生成的 token
            # 之前的 token 都已经算过并存在 KV Cache 中了，绝对不能重复计算
            input_ids.append(seq.last_token)
            # 确定位置，比如现在是第 101 个字，Position 就是 100
            # 模型需要这个数字来计算 RoPE（旋转位置编码）
            positions.append(len(seq) - 1)
            # 告诉 Attention 算子，本次 Attention 计算需要回头去 KV Cache 中看 len(seq) 这么长的数据
            context_lens.append(len(seq))
            # 算出这新的 token 的 KV 数据应该存到显存的哪个物理地址
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).npu(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).npu(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).npu(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        # 准备页表
        block_tables = self.prepare_block_tables(seqs)
        # 设置上下文
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            # 收集每一个 Sequence 对象
            # 每一个 Sequence 设定的温度很可能是不同的
            temperatures.append(seq.temperature)
        # 将所有的温度打包成一个 Tensor，并且使用 pin_memory 和 non_blocking 这样的性能优化方式
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).npu(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        logger.info(f"{'prefill' if is_prefill else 'decode'} execute tokens: {len(input_ids)}")
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Eager Mode
            # - prefill 阶段，因为 prefill 的输入的形状不固定，动态性比较强，不适合使用固定形状的 Graph
            # - enforce_eager：强制关闭优化，通常用于 Debug 模式
            # - BatchSize>512，大 Batch 下，GPU 忙不过来，Graph 的加速收益递减
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph Mode
            # Decode 阶段，输入形状比较固定，seq_len 永远是 1
            # 唯一的变量是 batch_size，也就是有多少用户在同时生成
            bs = input_ids.size(0)
            context = get_context()

            # 1. 选图（Graph Selection）
            # 我们可能录制了 BatchSize=1,2,4,8 的多个图
            # 如果当前来了 3 个请求，我们就只能找一个比 3 大的图，比如用 BatchSize=4 的图
            # 这叫 Padding Batch。
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            # 2. 填坑（Filling the Placeholders）
            # graph_vars 是我们在录制图时预留的固定显存地址
            # 我们必须把当前的数据（input_ids, positions）拷贝到这些固定的地址中去
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # 填 slot_mapping（物理显存地址）
            # 先填 -1 清空旧数据，防止越界访问，再填入有效数据
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            # 播放 CUDA Graph
            graph.replay()
            # 取货：从预设的输出地址中将结果取出来，只取前 Batch Size 个有效数据
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 数据准备：这一步在所有的 Rank(0, 1, 2, 3) 上都会执行
        # 无论是 Driver 还是 Worker，都需要计算出 slot_mapping、block_tables 等元数据
        # 并且通过 set_context 设置好全局上下文，告诉 GPU 数据在哪里
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        # 只有 rank0 需要关心如何采样
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 所有的 logits 开始进行神经网络计算
        logits = self.run_model(input_ids, positions, is_prefill)
        # rank0 进行选词
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # 1. 确定天花板（Defining Limits）
        # 我们支持多大的 Batch Size？（比如说 256）
        max_bs = min(self.config.max_num_seqs, 512)
        # 一个序列最多能有多少个 Block（比如说 4096 / 16 = 256 个）
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 2. 申请占位符（Place Holders）
        # 这些 Tensor 是永久驻留在显存中的
        # 无论后面跑 BatchSize=1 还是 BatchSize=8，我们都只是在读写这些 Tensor 的 slice
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        # 页表是二维的：[BS, MaxBlocks]
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        # 输出也是预分配好的
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 3. 定义档位（Batch Size Buckets）
        # 小的 Batch 逐个录（1， 2， 4，8），大的 Batch 按 16 步进
        # 这样能平衡图的数量（显存占用）和 Padding 的浪费
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 倒序录制
        # 因为最大的 Batch Size 占用的显存最多。先录最大的，可以让 Pytorch 的分配器把
        # 最大的显存块先申请下来，后面的小图就可以复用这块显存（通过 graph_tool）
        # 避免显存碎片化和重复申请
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            # 设置 context，本次录制使用的是前面申请的那些静态 Tensor 的前 batch size 个元素
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )

            # warmup 热身
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup

            # 正式录制：self.graph_pool 保证了所有图共享同一块私有显存池
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture

            # 如果是第一次（最大的那个 Batch Size），就把它的内存池保存下来，给后面的小图复用
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存操作句柄
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
