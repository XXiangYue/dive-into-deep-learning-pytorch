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
