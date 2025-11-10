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


def can_broadcast(shape_a: tuple, shape_b: tuple) -> bool:
    """从尾部右对齐逐维比较：相等或其中一个是 1 就能广播。"""
    for da, db in zip(reversed(shape_a), reversed(shape_b)):
        if da != db and da != 1 and db != 1:
            return False
    return True


def demo_broadcast() -> None:
    """跑几组形状，看看哪些能广播、结果形状是什么。"""
    cases = [
        ((3, 1), (1, 2)),
        ((3,), (3, 2)),
        ((3, 1), (3, 2)),
        ((2, 3, 4), (4,)),
        ((2, 3, 4), (3, 1)),
        ((5,), (2, 3)),
    ]
    for sa, sb in cases:
        ok = can_broadcast(sa, sb)
        note = ""
        if ok:
            a, b = torch.ones(sa), torch.ones(sb)
            note = f"-> {(a + b).shape}"
        print(f"{str(sa):<14} + {str(sb):<10} broadcast={str(ok):<6} {note}")
