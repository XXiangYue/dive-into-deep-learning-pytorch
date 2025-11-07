"""环境自检：在动手之前先把「能不能跑」这件事确认掉。

只检查四件事，不训练任何模型：
1. Python / torch / numpy 版本与设备信息；
2. 自动微分是否真的在工作（用 f = 3x^2 的手算导数对照）；
3. 随机种子的设置接口是否齐全，以及同一 seed 两次采样是否真的一致；
4. 张量在 CPU/GPU 之间搬一次之后 dtype 有没有被悄悄改掉。

用法：python code/01_env_check.py

2025-11-07 修订：
- 原来 set_seed 只固定了 torch.manual_seed，有 GPU 的机器上还是不一致，
  补 torch.cuda.manual_seed_all 和 cudnn.deterministic；
- 加了 check_determinism，把「固定种子」这件事从「我设了」变成「我验证了」。
"""

import platform
import random
import sys

import numpy as np
import torch

# 全书统一随机种子。后面每章都从这里抄，方便对齐 run-log 里的数字。
SEED = 42


def describe_platform() -> dict:
    """收集版本与设备信息，返回字典，方便打印或写进 run-log。"""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        # 注意是 torch.version.cuda，写成 torch.version 会 AttributeError
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


def set_seed(seed: int = SEED) -> None:
    """固定所有随机源。CPU 上没有 cuda 分支，所以都加了 is_available 判断。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 卷积算法固定下来，代价是慢一点，换可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_autograd() -> dict:
    """最小可验证的自动微分：y = sum(3x^2)，dy/dx 应为 6x。"""
    x = torch.arange(4.0, requires_grad=True)
    y = (3 * x ** 2).sum()
    y.backward()
    analytic = 6 * x.detach()
    return {
        "x": x.detach().tolist(),
        "grad": x.grad.tolist(),
        "analytic": analytic.tolist(),
        # 用 allclose 而不是 ==，浮点乘法顺序不同就会有 1e-7 级差异
        "allclose": bool(torch.allclose(x.grad, analytic)),
    }


def check_determinism(repeat: int = 2) -> bool:
    """同一 seed 连续采样 repeat 次，检查结果是否逐个相等。

    只固定种子不验证，等于没验证：中间可能有别的地方消耗了随机数流。
    """
    samples = []
    for _ in range(repeat):
        set_seed()
        samples.append(torch.randn(3, 4))
    return all(torch.equal(samples[0], other) for other in samples[1:])


def check_device_roundtrip() -> dict:
    """张量搬到计算设备再搬回来，dtype 和数值都应保持不变。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    y = x.to(device).clone()
    return {
        "device": str(device),
        "dtype_kept": y.dtype == x.dtype,
        "value_kept": bool(torch.equal(y.cpu(), x)),
    }


if __name__ == "__main__":
    for key, value in describe_platform().items():
        print(f"{key:>15}: {value}")

    result = check_autograd()
    print("x        :", result["x"])
    print("x.grad   :", result["grad"])
    print("6x       :", result["analytic"])
    print("grad 正确:", result["allclose"])

    print("seed 可复现:", check_determinism(repeat=2))
    print("设备往返  :", check_device_roundtrip())
