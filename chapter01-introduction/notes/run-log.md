# 第 1 章 运行记录

环境：Windows 11 / Python 3.12.10 / torch 2.4.1+cpu / numpy 1.26.4，无 GPU。
所有命令都在仓库根目录执行。

## 2025-11-07 21:10 环境自检

```bash
python code/01_env_check.py
```

输出（原样粘贴）：

```text
        python: 3.12.10
      platform: Windows-11-10.0.22631-SP0
         torch: 2.4.1+cpu
         numpy: 1.26.4
 cuda_available: False
   cuda_version: None
   device_count: 0
x        : [0.0, 1.0, 2.0, 3.0]
x.grad   : [0.0, 6.0, 12.0, 18.0]
6x       : [0.0, 6.0, 12.0, 18.0]
grad 正确: True
```

耗时 0.9 秒（大头是 import torch）。
`cuda_version` 是 `None`，因为装的是 CPU 版；`torch.version.cuda` 不会报错，
只有 `torch.version` 会。这一点记一下。

## 2025-11-07 21:26 张量演示

```bash
python code/02_first_tensor.py
```

关键几行：

```text
arange    shape=(12,)        dtype=torch.int64 numel=12   device=cpu
a: (3, 1) + b: (1, 2) -> c: (3, 2)
tensor([[0, 1],
        [1, 2],
        [2, 3]])
x.grad   : [0.0, 2.0, 4.0, 6.0]
没清零再 backward: [0.0, 4.0, 8.0, 12.0]
清零后 backward  : [0.0, 2.0, 4.0, 6.0]
```

「没清零再 backward」那行把旧梯度翻了一倍，正好说明了 `.grad` 是在累加，
不是覆盖。这是我这次最想验证的一点，验证到了。

## 2025-11-07 09:40 计划表

```bash
python code/03_plan_tracker.py
```

表格只贴前 8 行和最后 2 行（中间是 11~14 章，格式一样）：

```text
| 章 | 目录 | 工时 | 起始周 | 占用周数 |
|:---:|---|---:|---:|---:|
| 1 | chapter01-introduction | 4 | 1 | 1 |
| 2 | chapter02-preliminaries | 12 | 1 | 2 |
| 3 | chapter03-linear-neural-networks | 14 | 2 | 2 |
| 4 | chapter04-multilayer-perceptrons | 16 | 4 | 2 |
| 5 | chapter05-deep-learning-computation | 10 | 5 | 1 |
| 6 | chapter06-convolutional-neural-networks | 18 | 6 | 2 |
| 7 | chapter07-modern-convolutional-neural-networks | 22 | 8 | 3 |
| 8 | chapter08-recurrent-neural-networks | 16 | 10 | 2 |
...
| 15 | chapter15-natural-language-processing-applications | 14 | 21 | 2 |
```

汇总：

```text
             chapters: 15
          total_hours: 218.0
      weeks_to_finish: 22
avg_hours_per_chapter: 14.53
      longest_chapter: (7, 22.0)
             heaviest: [(7, 22.0), (10, 20.0), (6, 18.0)]
```

第一版的 `build_plan` 是「一章塞一周、放不下就顺延」，跑出来只有 15 周，
明显是假的——第 7 章 22 小时怎么可能一周读完。改成按累计工时推算起始周之后，
`weeks_to_finish` 是 $\lceil 218/10 \rceil = 22$ 周，第 7 章从第 8 周铺到第 10 周，
这是全表唯一跨三周的一章。

总工时 218 小时，比我一开始估的 150 小时多了 45%。
这一条有实际影响：原计划「每天 1 小时、五个月读完」正好 150 小时，现在不够。
两个选择——提到每周 12 小时（218/12 ≈ 18 周），或者把第 12 章（计算性能，8 小时）先放着。
我选了后者，因为我没有多卡环境，读第 12 章只能读个概念，收益最低。

第 7 章（现代 CNN）22 小时是全书最重的一章，AlexNet 到 DenseNet 七个网络，
每个都要手写一遍，我打算在那里多留一周 buffer。
