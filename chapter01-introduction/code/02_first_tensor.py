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


def demo_broadcast() -> torch.Tensor:
    """广播：两个张量按尾部维度右对齐，长度为 1 的维度被复制到对方长度。"""
    a = torch.arange(3).reshape(3, 1)   # (3, 1)
    b = torch.arange(2).reshape(1, 2)   # (1, 2)
    c = a + b                           # 结果是 (3, 2)
    # 我原来以为 a + b 会报错，其实两条长度为 1 的轴各自被撑开了
    print("a:", tuple(a.shape), "+ b:", tuple(b.shape), "-> c:", tuple(c.shape))
    print(c)
    return c


def demo_autograd() -> None:
    """对 y = x^T x 求导，梯度应该是 2x；顺便验证梯度会累加。"""
    x = torch.arange(4.0, requires_grad=True)
    y = torch.dot(x, x)
    y.backward()
    print("x        :", x.detach().tolist())
    print("x.grad   :", x.grad.tolist())
    print("2x       :", (2 * x.detach()).tolist())

    # 不清零就再 backward，梯度会加到旧值上，这是最高频的坑之一
    (2 * x).sum().backward()
    print("没清零再 backward:", x.grad.tolist())

    x.grad.zero_()
    (2 * x).sum().backward()
    print("清零后 backward  :", x.grad.tolist())
