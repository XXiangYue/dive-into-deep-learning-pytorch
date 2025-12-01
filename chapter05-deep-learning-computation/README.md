# 第 5 章 深度学习计算

对应原书：`chapter_deep-learning-computation`（5.1 ~ 5.7）。这一章没有新算法，
全是工程：怎么定义层和块、参数怎么管、模型怎么存、代码怎么写才能在有卡没卡上都能跑。

我把这一章当成一次对自己代码的重构。前四章我一直在用 list 装参数、手写训练循环，
能跑但没法维护；这一章之后全部换成 `nn.Module` + `nn.Parameter` + `state_dict`。

## 本章目标

1. 说清一个块（block）的三个职责：存参数、算前向、报结构；
2. 能用两种方式写自定义层：无参层（如 `CenteredLayer`）和带参层（如 `MyLinear`）；
3. 知道什么写法会让参数「消失」，并用 `named_parameters()` 自查；
4. 会用 `state_dict` 存档，理解它和 `torch.save(整个模型)` 的区别；
5. 写出设备无关的代码，在没有 GPU 的机器上走 `cpu` 分支也能跑通。

## 知识点清单

| 小节 | 核心 | 我的检查标准 |
|---|---|---|
| 5.1 层和块 | `nn.Module`、`add_module`、容器 | 能自己实现一个 `MySequential` |
| 5.2 参数管理 | `parameters()` / `named_parameters()` | 能解释 `[Parameter]` 为什么不注册 |
| 5.3 延后初始化 | 浅层 `LazyLinear`、`reset_parameters` | 知道第一次前向才建参数 |
| 5.4 自定义层 | 无参层、带参层、`apply` 初始化 | 能写出正确的形状和初始化 |
| 5.5 读写文件 | `state_dict`、`load_state_dict` | 能说清为什么存字典不存对象 |
| 5.6 / 5.7 GPU | `to(device)`、`map_location` | 能写出全流程设备无关的代码 |

## 产出物

| 文件 | 作用 |
|---|---|
| `code/01_custom_layers.py` | `CenteredLayer`、`MyLinear`、`MySequential`、`MLPBlock` |
| `code/02_param_management.py` | 嵌套块的参数收集、初始化、参数共享 |
| `code/03_state_dict_io.py` | `state_dict` 存取（内存里做往返，不落盘） |
| `code/04_device.py` | 设备无关的模型与数据迁移、几步训练 |
| `notes/run-log.md` | 参数个数、初始化前后的 std、存档字节数、设备信息 |
| `exercises/ch05-exercises.md` | 6 道题，含「参数为什么会消失」的机制 |

## 环境说明

这台机器没有 GPU（`torch.cuda.is_available()` 是 False），所以 5.6/5.7 节的代码
只能走 `cpu` 分支。我把「有卡时会走哪些行」在注释里标出来，
`code/04_device.py` 保证两种情况都不需要改代码。

## 参考小节

5.1 层和块 · 5.2 参数管理 · 5.3 延后初始化 · 5.4 自定义层 ·
5.5 读写文件 · 5.6 GPU · 5.7 小结。

预算 10 小时，实际 9 小时。这一章读起来最快，因为全是我写代码时想问的问题。
