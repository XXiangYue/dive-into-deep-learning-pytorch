"""张量基础练习：形状变换、广播、按轴求和、原地操作。

这个脚本没有训练，全部输出都是为了建立形状直觉。
每个函数都能单独 import 调用，方便在别的章节里复用 shape_report。
"""

import torch


def shape_report(name: str, x: torch.Tensor) -> str:
    """返回一行形状描述，比直接 print(x) 更适合看大张量。"""
    line = f"{name:<12} shape={str(tuple(x.shape)):<12} dtype={str(x.dtype):<14} " \
           f"numel={x.numel()}"
    print(line)
    return line


def demo_create_and_reshape() -> None:
    """创建 + 形状变换：reshape / view / transpose 的区别。"""
    x = torch.arange(24)
    shape_report("arange(24)", x)
    shape_report("reshape(2,3,4)", x.reshape(2, 3, 4))
    shape_report("reshape(2,-1)", x.reshape(2, -1))
    shape_report("reshape(4,6).T", x.reshape(4, 6).T)
    # squeeze/unsqueeze 只增删长度为 1 的轴，不改变数据排布
    shape_report("unsqueeze(0)", x.unsqueeze(0))
    shape_report("squeeze", x.unsqueeze(0).squeeze(0))
