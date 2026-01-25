from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        # block_id 是 block 的唯一标识符，通常直接映射到 GPU 上预先分配好的 KV Cache 张量的索引
        self.block_id = block_id
        # 引用计数，表示当前有多少个 Sequence 正在使用这个 block
        self.ref_count = 0
        # 用于前缀缓存。根据 block 内部的 token_ids 生成一个哈希值，方便后续的请求复用 block
        self.hash = -1
        # 记录这个 block 存储了哪些 token
        self.token_ids = []

    # 当这个 block 被分配给某个 sequence，并且填入了具体的 token 数据之后调用
    # 功能：更新元数据（哈希值和 token 内容）
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 当 Block Manager 从空闲池中拿到一个新的 block 准备分配给一个新的 sequence 时被调用
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size

        # 预先创建好所有的 Block 对象，
        # 这对应 GPU 上预先分配好的一大块显存池
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # 哈希查找表
        # 作用：用于 Prefix Caching（前缀缓存）
        # 逻辑：Key 是内容的哈希值，Value 是对应的 block_id
        # 比如：我想找存了 “Hello World” 的 Block，查表发现 block_id = 5 里面有，那就可以直接复用
        self.hash_to_block_id: dict[int, int] = dict()

        # 空闲链表
        # deque（双端队列）用作栈或者队列，存放当前没人用的 block_id
        # 初始状态下，所有块都是空闲的
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # 记录哪些块正在干活儿，初始状态下，所有块都是空闲的
        self.used_block_ids: set[int] = set()

    # classmethod 表示这是一个类方法，而不是实例方法
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 使用 xxhash 来计算哈希值，这是一种计算比较快的哈希方法
        h = xxhash.xxh64()

        # 链式哈希（Chaining）
        # 如果这不是序列的第一个块，我们就需要把前一个块的哈希值（prefix）加进来
        # 这样，第 N 个块的哈希值就不仅仅取决于它自己的 Token，还取决于它前面的历史
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        # 加入当前的 Token 数据
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 分配指定块
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        # 安全检查：只有引用计数为 0 （没人用）的块，才能被重新分配
        assert block.ref_count == 0

        # 激活块
        # 这里调用了 Block 中的 reset()，将 ref_count 设置为 1
        block.reset()

        # 从空闲表中移出，加入使用表
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        # 安全检查：必须确保引用计数归零了才能彻底回收
        # 如果 ref_count > 0，说明还有其他的 sequence 正在共享这个块，那就不能回收物理显存
        assert self.blocks[block_id].ref_count == 0

        # 从已用表中移出，加入空闲表
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 检查当前的 BlockManager 中是否有足够的空闲的块
    # 来满足一个新的 sequence 的初始内存分配请求
    def can_allocate(self, seq: Sequence) -> bool:
        # 判断：当前空闲块的数量 >= 序列所需的块数量
        return len(self.free_block_ids) >= seq.num_blocks

    # 核心逻辑：一段一段地处理 Token，只要前面的能匹配上缓存，就继续复用
    # 一但断了，后面就必须全部重新分配新块
    def allocate(self, seq: Sequence):
        # 确保这个序列还没有被分配过新块（是新来的）
        assert not seq.block_table

        # 当前已经累积的哈希值
        h = -1
        # 标记是否已经发生了缓存未命中
        cache_miss = False

        # 遍历序列需要的每一个块（Block）
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # 获取当前这一个块里面的 Token 数据

            # 1. 计算哈希（尝试匹配指纹）
            # 只有满的块才计算哈希（通常只有最后一块可能不满，不满的不缓存）
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )

            # 2. 查表
            # 看看有没有哪个块的指纹跟当前的一样
            block_id = self.hash_to_block_id.get(h, -1)

            # 3. 验证缓存有效性
            # 如果表里面没有查到（block_id == -1）
            # 或者查到了但是内容不对（哈希碰撞的情况，概率极低）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            # 4. 分支判断：复用旧块还是开新块
            if cache_miss:
                # 没有命中
                # 从空闲池子里拿一个新的块 ID
                block_id = self.free_block_ids[0]
                # 物理分配它（标记为 Used，ref_count = 1）
                block = self._allocate_block(block_id)
            else:
                # 命中缓存
                # 记录节省了多少计算量
                seq.num_cached_tokens += self.block_size

                if block_id in self.used_block_ids:
                    # 这个块正在被别人使用的情况，我们让它的引用计数 +1
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 这个块虽然在哈希表里面，但是目前是空闲状态
                    # 说明它之前被释放过，但是没有被新的数据覆盖
                    # 我们直接把它从空闲池里面捞回来"复活"，不需要重新计算
                    block = self._allocate_block(block_id)

            # 5. 更新元数据
            if h != -1:
                # 更新块的指纹和内容
                block.update(h, token_ids)
                # 更新哈希表
                self.hash_to_block_id[h] = block_id

            # 6. 将块 ID 加入序列的页表
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        # 倒序遍历这个 sequence 持有的所有 Block ID
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            # 引用计数减一
            block.ref_count -= 1
            # 真正的物理释放
            # 只有当引用计数降为 0 时，才能把它放回 free_block_ids
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 清理 sequence 自身的状态
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # 如果 len(seq) % self.block_size == 1，说明刚才那个 Token 是一个新的一块的“第一个 token”
        # 这意味着我们需要一个新的物理块（Block）
        # 此时，必须保证 free_block_ids 里面至少有一个空闲块。

        # 如果不是 == 1（比如说 == 2,3,0），说明我们还在当前的 Block 中填空
        # 不需要新的 Block，所以总是返回 True（因为需要的空闲块数量 >= 0）
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        # 获取当前序列持有的最后一个 Block 对象
        last_block = self.blocks[block_table[-1]]

        # 情况 A：刚跨入新的一块（Start of a new Block）
        # 比如 Block 大小是 4，现在长度变成了 5、9、13...
        if len(seq) % self.block_size == 1:
            # 检查上一块的状态
            # 因为我们刚刚迈入新的一块，说明“上一块”肯定已经满了
            # 既然满了，它在之前的步骤里肯定已经被计算过哈希了
            assert last_block.hash != -1

            # 分配新块
            # 从空闲池拿一个新的 ID
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)

            # 挂载到页表
            block_table.append(block_id)

        # 情况 B：刚好填满一块（End of a block）
        # 比如 Block 大小是 4，此时长度变成了 4、8、12...
        elif len(seq) % self.block_size == 0:
            # 检查状态
            # 刚填满还没来得及处理，所以哈希应该是初始值 -1
            assert last_block.hash == -1

            # 准备计算哈希
            # 拿到这块里所有的 Token
            token_ids = seq.block(seq.num_blocks - 1)

            # 获取前缀哈希（Prefix Hash）
            # 链式哈希的关键：当前的哈希 = Hash(前一块的哈希 + 当前块的内容)
            # block_table[-2] 就是倒数第二个块（前一块）
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1

            # 计算并封存哈希
            h = self.compute_hash(token_ids, prefix)

            # 更新 Block 对象，打上最新的哈希
            last_block.update(h, token_ids)

            # 注册到全局哈希表
            # 从这一刻起，其他的 sequence 如果前缀和这个一样，就可以复用这个块了
            self.hash_to_block_id[h] = last_block.block_id

        # 情况 C：中间状态（Middle of a block）
        # 比如 Block 大小是 4，此时长度变成了 2、3、6、7...
        else:
            # 还没满，不能算哈希，只能是“脏”状态（-1）
            assert last_block.hash == -1
