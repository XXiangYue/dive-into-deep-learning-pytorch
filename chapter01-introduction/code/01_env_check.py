"""环境自检：在动手之前先把「能不能跑」这件事确认掉。

只检查三件事，不训练任何模型：
1. Python / torch / numpy 版本与设备信息；
2. 自动微分是否真的在工作（用 f = 3x^2 的手算导数对照）；
3. 随机种子的设置接口是否齐全（后续每章都要用）。

用法：python code/01_env_check.py
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
    """固定 Python / numpy / torch 三处随机源。

    这三个是彼此独立的随机数发生器，只固定 torch 的话，
    数据打乱（用 random）和 numpy 采样还是每次不一样。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
