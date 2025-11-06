"""张量初体验：把「多维数组 + 广播 + 自动微分」三件事各跑一遍。

第 1 章不做训练，只建立手感。三个函数分别演示：
形状变换、广播规则、requires_grad 与 backward 的基本用法。
"""

import torch


def describe_tensor(name: str, x: torch.Tensor) -> None:
    """打印张量的形状、dtype、元素个数与所在设备。"""
    print(
        f"{name:<9} shape={str(tuple(x.shape)):<12} "
        f"dtype={str(x.dtype):<8} numel={x.numel():<4} device={x.device}"
    )
