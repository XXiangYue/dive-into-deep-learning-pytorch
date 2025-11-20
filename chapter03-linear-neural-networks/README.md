# 第 3 章 线性神经网络

对应原书：`chapter_linear-networks`（3.1 ~ 3.7）。这一章是全书的分水岭：
第 2 章的东西在这一章第一次拼成「能训练的模型」，后面所有网络都是它的加层版。

## 本章目标

1. 从零实现线性回归：造数据、小批量、手写 SGD、看 loss 逐 epoch 下降；
2. 用 `nn.Linear` + `MSELoss` + `optim.SGD` 写简洁版，对照两者差在哪；
3. 搞清 softmax 回归：为什么用 one-hot、为什么配交叉熵、梯度为什么是 $\hat{y} - y$；
4. 至少返工一次（这一章我返工了两次：SGD 的原地操作、softmax 的数值溢出）。

## 知识点清单

| 小节 | 核心 | 我的检查标准 |
|---|---|---|
| 3.1 线性回归 | 模型、平方损失、解析解、小批量 SGD | 能手推 $\partial L/\partial \mathbf{w}$ |
| 3.2 从零实现 | `data_iter`、`sgd`、训练循环骨架 | 能不看书写出训练循环 |
| 3.3 简洁实现 | `nn.Sequential`、`DataLoader`、`optim` | 能说清 `MSELoss` 已经除过 batch |
| 3.4 softmax 回归 | one-hot、交叉熵、准确率 | 能解释为什么不直接用 MSE |
| 3.5 图像分类数据集 | 标签是类别下标，不是 one-hot | 能说出数据集该怎么造 |
| 3.6 从零实现 softmax | 向量化实现、按行归一化 | 能写出不带 for 的 softmax |
| 3.7 简洁实现 | `CrossEntropyLoss` 内含 softmax | 知道它期望 logits 而不是概率 |

## 产出物

| 文件 | 作用 |
|---|---|
| `code/01_linear_regression_scratch.py` | 线性回归从零：合成数据 + 手写 SGD |
| `code/02_linear_regression_concise.py` | 线性回归简洁版，与 01 对照参数量与损失 |
| `code/03_softmax_scratch.py` | softmax 回归从零：三类高斯团 + 交叉熵 |
| `notes/run-log.md` | 线性回归 5 epoch 的 loss、学习率扫描、softmax 的 nan 记录 |
| `exercises/ch03-exercises.md` | 6 道题，含 softmax 梯度推导 |

## 关于数据

这一章**不下载 Fashion-MNIST**。原因有三：
Fashion-MNIST 有 30MB，宿舍网不稳定；它的图像是 28×28，而这一章的重点是
「多类 + 交叉熵」这套机制，跟图像尺寸无关；合成的高斯团可以随时改类别数和噪声，
做消融实验更快。等第 6 章讲卷积时再换成真实图像数据集。

## 参考小节

3.1 线性回归 · 3.2 线性回归的从零开始实现 · 3.3 线性回归的简洁实现 ·
3.4 softmax 回归 · 3.5 图像分类数据集 · 3.6 softmax 回归的从零开始实现 ·
3.7 softmax 回归的简洁实现。

预算 14 小时（计划表里的数字），实际用了 16 小时，超的两小时全花在两个 bug 上。
