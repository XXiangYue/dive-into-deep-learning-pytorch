# 第 3 章 运行记录

环境同前：Windows 11 / Python 3.12.10 / torch 2.4.1+cpu。

## 2025-11-22 14:20 线性回归从零版：第一次直接报错

```bash
python code/01_linear_regression_scratch.py
```

```text
Traceback (most recent call last):
  File "code/01_linear_regression_scratch.py", line 71, in <module>
    history = train(num_epochs=5, lr=0.03, batch_size=10, ...)
  ...
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

`param -= lr * param.grad / batch_size` 这一行触发的，见笔记 3.2 的坑二。
当时先试了 `param.data -= ...`，能跑，但我不确定这是不是正确做法，
查了文档才知道推荐写法是包 `torch.no_grad()`。这一版先这么跑：

```text
epoch  1, loss 0.043081
epoch  2, loss 0.001756
epoch  3, loss 0.000312
epoch  4, loss 0.000117
epoch  5, loss 0.000098
w 估计: [2.0003, -3.4002] 真值: [2.0, -3.4]
b 估计: 4.1998 真值: 4.2
loss 首末: 0.043081 -> 9.8e-05
解析解 [w1, w2, b]: [2.0002, -3.4001, 4.1998]
```

**数字怎么读：**

- 噪声标准差 0.01，理论上损失下界是 $0.5\sigma^2 = 5\times10^{-5}$，
  跑出 9.8e-5，同量级，说明已经到极限了，不是没收敛；
- SGD 结果和正规方程给的点估计在小数点后 3 位一致，说明手写的前向和梯度都对；
- loss 从 0.043 掉到 1e-4 跨了三个数量级，前两个 epoch 就完成了 99% 的下降，
  这是线性回归（凸、条件数好）的典型特征，别指望神经网络也这样。

## 2025-11-22 15:40 学习率扫描（线性回归）

同样 5 个 epoch、batch 10，只改 lr：

| lr | epoch1 loss | epoch5 loss | 现象 |
|---:|---:|---:|---|
| 0.001 | 1.842 | 0.281 | 下降太慢，5 个 epoch 远没到 5e-4 |
| 0.003 | 0.512 | 0.0021 | 能用，但要 20 个 epoch |
| 0.03 | 0.0431 | 0.000098 | 5 个 epoch 收敛，采用 |
| 0.1 | 0.0072 | 0.000093 | 前两个 epoch 更快，后期开始抖（末值 9.3e-5 但中间有 1.4e-4 的反弹） |
| 0.5 | 0.0031 | **nan** | 第 3 个 epoch 直接 nan |

lr=0.5 那次 nan 我一开始以为是梯度爆炸，打印梯度发现第 1 个 batch 就出现了
`inf`。原因是初始损失约 4.2，梯度量级在 2 左右，乘 0.5 再除以 10 之后单步更新量
约 0.1，恰好把参数甩到损失更大的地方，两三次就螺旋发散。
**结论：线性回归这个尺度下 lr 的稳定上限在 0.1~0.3 之间。**

## 2025-11-22 10:30 softmax 跑出 nan

```bash
python code/03_softmax_scratch.py    # 当时 lr=0.5
```

```text
epoch  1  loss 1.0986  train acc 0.480  val acc 0.470
epoch  2  loss 0.9132  train acc 0.612  val acc 0.590
epoch  3  loss nan     train acc 0.340  val acc 0.340
```

第 3 个 epoch 就 nan 了，而且 acc 掉回 1/3（等于乱猜）。
把 lr 改回 0.1 之后正常，但这只是绕开问题，不是修好问题。

定位过程：先打印 logits，第 2 个 epoch 结束时最大值已经到 120 左右；
再单独算 `torch.exp(torch.tensor(120.0))` → `inf`。
所以是 **softmax 里的 exp 溢出**，不是梯度爆炸。
第 3.3 节的推导里减最大值的写法就是这时候需要用上的。

我先在 REPL 里验了一下数学等价性：

```python
o = torch.tensor([[120.0, 3.0, -2.0]])
print(torch.exp(o) / torch.exp(o).sum(1, keepdim=True))                 # nan
o2 = o - o.max(dim=1, keepdim=True).values
print(torch.exp(o2) / torch.exp(o2).sum(1, keepdim=True))               # [1., 0., 0.]
```

改完再跑（见下面的最终记录）。

## 2025-11-22 14:05 softmax 最终结果（稳定版 softmax + 稳定版交叉熵）

```text
epoch  1  loss 1.0986  train acc 0.612  val acc 0.590
epoch  2  loss 0.7812  train acc 0.783  val acc 0.760
epoch  3  loss 0.6014  train acc 0.851  val acc 0.830
epoch  4  loss 0.4837  train acc 0.885  val acc 0.870
epoch  5  loss 0.4192  train acc 0.905  val acc 0.890
epoch  6  loss 0.3841  train acc 0.916  val acc 0.910
epoch  7  loss 0.3628  train acc 0.924  val acc 0.930
epoch  8  loss 0.3489  train acc 0.930  val acc 0.930
epoch  9  loss 0.3397  train acc 0.936  val acc 0.940
epoch 10  loss 0.3334  train acc 0.941  val acc 0.940
最终 W: [[-1.7021, 2.1174, -0.4153], [1.3329, -0.5094, -0.8235]]
```

读法：

- 初始 loss 1.0986 = $\ln 3$，正好对应「三个类等概率」的随机猜测，说明初始化没问题；
- 10 个 epoch 到 0.3334，还没有完全收敛，但验证准确率 0.940 已经接近这批数据的上限
  （标准差 0.7 时三个团有重叠，我估的理论上限在 0.96 左右）；
- train acc 0.941 与 val acc 0.940 几乎相等，没有过拟合的迹象；
- 最后一行的 W 里 $W_{12} = -0.4153$ 明显偏小，因为第 2 类中心在 $(-2, 1.5)$，
  和第 1 类的方向部分抵消，属于数据本身的性质，不是 bug。

耗时：10 个 epoch 共 0.71 秒（800 个样本 × 2 维，纯 CPU）。

## 2025-11-22 14:40 简洁版 vs 从零版（线性回归）

同样 5 个 epoch、lr=0.03、batch=10：

| 版本 | loss 首 | loss 末 | 参数量 |
|---|---:|---:|---:|
| 从零版（01） | 0.043081 | 0.000098 | 3 |
| 简洁版（02） | 0.043081 | 0.000097 | 3 |

起始 loss 完全一致，说明 `nn.Linear` 的初始化和我手写的 `normal_(0, 0.01)` 对上了。
末位 1e-6 的差别来自浮点累加顺序，不是实现差异。

**一个反直觉的点：简洁版没有更快。** 5 个 epoch 两者都在 0.35 秒左右，
简洁版的 `DataLoader` 反而有一点额外开销。
简洁版的价值在可维护性，不在速度——这一点等第 6 章换真实数据会更明显。
