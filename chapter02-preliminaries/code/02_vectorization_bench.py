"""向量化 vs for 循环：用自己机器上的数字验证第 1 章那句话。

任务故意选最简单的逐元素加法：
- naive 版按行遍历，每次只处理一行（模拟「不想清楚就写 for」的写法）；
- vectorized 版直接 a + b。
形状取 4096 x 1024 的 float32，约 4M 个元素，正好在 CPU 缓存装不下的量级。

2025-11-12 修订：
- 第一版用 time.time() 且没有热身，测到的 26ms 里七成是冷启动，见 run-log；
- 现在改成 perf_counter + 3 次热身丢弃 + 10 次取中位数；
- 顺手删掉了第一版里那两行没用的临时变量（get_num_threads / Event）。
"""

import statistics
import time

import torch

NUM_ROWS, NUM_COLS = 4096, 1024
WARMUP = 3
REPEAT = 10


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
    """计时一次，返回毫秒。perf_counter 分辨率比 time.time 高，且不受时钟调整影响。"""
    begin = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - begin) * 1000


def benchmark(fn, *args, warmup: int = WARMUP, repeat: int = REPEAT) -> dict:
    """热身丢弃 + 重复取中位数。返回中位数 / 最小值 / 样本数。"""
    for _ in range(warmup):
        fn(*args)
    times = [time_once(fn, *args) for _ in range(repeat)]
    return {
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "samples": len(times),
    }


def compare(repeat: int = REPEAT) -> list:
    """两个实现各测一遍，返回 [(名字, 统计字典)]。"""
    a, b = make_inputs()
    rows = []
    for name, fn in (("naive", add_naive), ("vectorized", add_vectorized)):
        rows.append((name, benchmark(fn, a, b, repeat=repeat)))
    return rows


def print_table(rows: list) -> None:
    """打印对比表，并给出加速比。"""
    print(f"{'impl':<12}{'median(ms)':>12}{'min(ms)':>10}{'max(ms)':>10}")
    for name, stat in rows:
        print(f"{name:<12}{stat['median']:>12.2f}{stat['min']:>10.2f}{stat['max']:>10.2f}")
    naive = {n: s for n, s in rows}["naive"]["median"]
    vec = {n: s for n, s in rows}["vectorized"]["median"]
    print(f"加速比: {naive / vec:.1f}x")


if __name__ == "__main__":
    a, b = make_inputs()
    ok = torch.allclose(add_naive(a, b), add_vectorized(a, b))
    print("两个实现结果一致:", ok)
    print("线程数:", torch.get_num_threads())
    print_table(compare())
