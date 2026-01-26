## Ascend NPU 环境快速安装

下面是一个最小可用的 Ascend NPU Python 环境配置和自测脚本，方便快速验证机器上的 NPU 是否能正常跑起来。

### 1. 创建 conda 环境

```bash
conda create -n ai-infra python=3.10
conda activate ai-infra
```

### 2. 安装 PyTorch 和依赖

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-npu==2.5.1
pip install "numpy<2.0.0"
pip install decorator scipy sympy cffi pyyaml pathlib2 psutil protobuf attrs transformers xxhash
```

### 3. 使用脚本测试 NPU 是否可用

把下面这段脚本保存为 `test_npu.py`（也可以直接在 Python 里敲），运行后如果能在 NPU 上完成加法运算，就说明环境基本 OK。

```python
import torch
import torch_npu

# 检查 NPU 是否真正可用
print(f"Is NPU available: {torch.npu.is_available()}")

# 尝试在 NPU 上做简单运算
try:
    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    z = x + y
    print("NPU computation successful!")
    print(z)
except Exception as e:
    print(f"Error: {e}")
```

运行：

```bash
python test_npu.py
```

### 4. 参考输出（示例）

正常情况下，你会看到类似下面的输出（数值本身会变）：

```text
Is NPU available: True
NPU computation successful!
[W125 19:54:38.403474684 compiler_depend.ts:137] Warning: Warning: Device do not support double dtype now, dtype cast repalce with float. (function operator())
tensor([[-0.1331,  0.2224],
        [-0.0730, -0.2272]], device='npu:0')
```

最后那条 warning 一般是精度相关的提示信息，正常情况下可以忽略。
