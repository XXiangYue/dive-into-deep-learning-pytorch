"""向量化 vs for 循环：用自己机器上的数字验证第 1 章那句话。

任务故意选最简单的逐元素加法：
- naive 版按行遍历，每次只处理一行（模拟「不想清楚就写 for」的写法）；
- vectorized 版直接 a + b。
形状取 4096 x 1024 的 float32，约 4M 个元素，正好在 CPU 缓存装不下的量级。
"""

import time

import torch

NUM_ROWS, NUM_COLS = 4096, 1024


def make_inputs(rows: int = NUM_ROWS, cols: int = NUM_COLS):
    """造两个连续存放的 float32 张量。"""
    a = torch.randn(rows, cols)
    b = torch.randn(rows, cols)
    return a, b


def add_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """逐行循环相加：Python 层要跑 rows 次，每次都进一次算子调度。"""
    out = torch.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = a[i] + b[i]
    return out


def add_vectorized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """一次算子调用完成全部元素。"""
    return a + b


def time_once(fn, *args) -> float:
    """计时一次，返回毫秒。"""
    begin = time.time()
    fn(*args)
    return (time.time() - begin) * 1000


def compare(repeat: int = 3) -> list:
    """两个实现各跑 repeat 次，返回 [(名字, 毫秒列表)]。"""
    a, b = make_inputs()
    rows = []
    for name, fn in (("naive", add_naive), ("vectorized", add_vectorized)):
        times = [time_once(fn, a, b) for _ in range(repeat)]
        rows.append((name, times))
    return rows


def print_table(rows: list) -> None:
    """打印对比表，并给出加速比。"""
    print(f"{'impl':<12}{'min(ms)':>10}{'max(ms)':>10}")
    for name, times in rows:
        print(f"{name:<12}{min(times):>10.2f}{max(times):>10.2f}")
    naive = min(t for n, t in rows if n == "naive")
    vec = min(t for n, t in rows if n == "vectorized")
    print(f"加速比: {naive / vec:.1f}x")


if __name__ == "__main__":
    a, b = make_inputs()
    ok = torch.allclose(add_naive(a, b), add_vectorized(a, b))
    print("两个实现结果一致:", ok)
    print("线程数:", torch.get_num_threads())
    print_table(compare(repeat=3))
