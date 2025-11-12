# 第 2 章 运行记录

环境同第 1 章：Windows 11 / Python 3.12.10 / torch 2.4.1+cpu / 8 线程 CPU。

## 2025-11-12 20:05 向量化 vs for 循环

```bash
python code/02_vectorization_bench.py
```

第一版输出（注意：这一版没有热身，且用了 `time.time()`）：

```text
两个实现结果一致: True
线程数: 8
impl            min(ms)   max(ms)
naive              26.41    168.30
vectorized          0.94      6.10
加速比: 28.1x
```

结论：4M 元素的逐元素加法，逐行 for 循环比向量化慢 28 倍。
naive 版慢在 Python 层的 4096 次算子调度，不在算术本身——两种写法做的浮点运算量完全一样。

但我对这个数字不放心，两个原因：
1. naive 的 min 和 max 差了 6.4 倍，第一次调用明显把内存分配、线程池初始化的开销算进去了；
2. 没有热身，第一次运行还包含 page fault 和缓存冷启动。

下一版加 3 次热身 + 取中位数再测一次。

## 2025-11-12 20:31 雅可比对照

```bash
python code/03_autograd.py
```

```text
手算雅可比:
 tensor([[2., 2.], [2., 1.], [0.5403, 0.0000]])
backward 逐行:
 tensor([[2.0000, 2.0000], [2.0000, 1.0000], [0.5403, 0.0000]])
手算 vs backward: True
手算 vs vjp     : True
x=2 时 dy/dx=12.0 (期望 12.0), d2y/dx2=12.0 (期望 12.0)
detach 后: {'x.grad': None}
```

`detach 后 x.grad 是 None` 这一条我特意留着：`z` 只连着 `u`，`u` 已经断图，
所以完全没有路径能回到 `x`，`.grad` 保持 None，不报错也不给警告。
这一点挺危险——断图断错了不会报错，只会静默不更新。
