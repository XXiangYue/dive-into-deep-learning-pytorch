# 第 2 章 预备知识 · 读书笔记

## 这章为什么不能跳

第 3 章从零实现线性回归的时候，会同时用到这章的至少五样东西：
`torch.normal` 造数据（2.6）、`@` 和 `sum` 写前向（2.3）、广播写损失（2.1）、
autograd 求梯度（2.5）、`zero_()` 清梯度（2.1 原地操作）。少一样就卡住。

## 依赖树（谁被后面哪一章用到）

```text
预备知识
├── 2.1 数据操作
│   ├── 张量创建：arange / zeros / ones / randn / tensor
│   ├── 形状：reshape / view / permute / transpose / squeeze
│   ├── 广播：尾部右对齐，长度为 1 的轴被复制
│   ├── 按轴运算：sum / mean / cumsum，以及 keepdim
│   └── 原地操作：add_ / += / zero_，与 id 不变
├── 2.2 数据预处理
│   ├── pandas 读表、缺失值（数值用均值，类别当一类）
│   ├── 类别特征 → 独热
│   └── 转张量：dtype 必须显式给 float32
├── 2.3 线性代数
│   ├── 标量 / 向量 / 矩阵 / 高阶张量
│   ├── 矩阵乘的两种视角（行加权、列组合）★
│   ├── 范数：L1 / L2 / Frobenius
│   └── 转置、对称矩阵、按轴求和
├── 2.4 微积分
│   ├── 导数、偏导、梯度
│   ├── 链式法则 ★
│   └── 为什么沿负梯度走能降函数值
├── 2.5 自动微分
│   ├── requires_grad / backward / .grad
│   ├── 计算图与反向累积（复杂度与深度同阶）
│   ├── detach：把子图从计算图里摘出来
│   ├── 控制流：梯度跟着实际走的分支
│   └── 二阶导：backward(create_graph=True)
└── 2.6 概率
    ├── 联合 / 条件 / 贝叶斯
    ├── 期望、方差、协方差
    ├── 均匀、正态、伯努利、多项
    └── 抽样与经验分布
```

标 ★ 的两处是我这一章要重点确认的：矩阵乘的视角决定后面能不能看懂 1×1 卷积，
链式法则决定能不能看懂反向传播。

## 我的读法

- 2.1、2.3 两节边读边在 REPL 里敲，形状随手 print；
- 2.4、2.5 两节先自己推一遍再对照 autograd 的结果，不一致就去查；
- 2.6 只求会用 `torch.normal` / `torch.multinomial`，概率论本身放到西瓜书里补。

## 2.1 数据操作：形状是第一公民

一个张量有三个属性要同时盯住：`shape`（形状）、`dtype`（元素类型）、`device`（在哪算）。
后面 90% 的报错都是这三者之一不匹配。最常见的两个：

- `expected scalar type Float but found Double` → numpy 默认 float64，torch 默认 float32；
- `Expected all tensors to be on the same device` → 有一个张量忘了 `.to(device)`。

### 广播规则（我自己的记法）

从**尾部**开始逐维比较，每一维必须满足「相等」或「其中一个是 1」，否则报错。
长度为 1 的那一维会被复制到对方的长度。

以 `a: (3, 1)` 和 `b: (1, 2)` 为例：

$$\underbrace{(3, 1)}_{\mathbf{a}} + \underbrace{(1, 2)}_{\mathbf{b}} \;\longrightarrow\; (3, 2)$$

第 2 维：$1$ 与 $2$ 比，$1$ 被撑成 $2$；第 1 维：$3$ 与 $1$ 比，$1$ 被撑成 $3$。
所以结果形状是 $(3, 2)$，实际计算的是

$$c_{ij} = a_{i} + b_{j}$$

这正好是「外积式」的加法表。理解成「两个一维向量做加法表」，比记规则好用。

### 容易错的地方

`a: (3,)` 和 `b: (3, 2)` 能不能广播？不能。`(3,)` 会被当成 `(1, 3)`，
尾部对齐后是 `(1,3)` vs `(3,2)`，第 2 维 $3 \ne 2$ 且都不是 1，报错。
想加就得显式 `a.reshape(3, 1)`。我第一次写线性回归的损失就栽在这里。

### 广播的代价与原地操作

广播不复制数据，只在计算时按需取元素，所以 `(3,1) + (1,2)` 不会真的先造两个 $3\times2$ 的中间张量。
但**结果**是一个新的 $3\times2$ 张量。

在训练循环里，这种「每次迭代新建一个大张量」会让内存不断申请释放。Python 的 `id()` 能看出来：

```python
before = id(Y)
Y = Y + X          # id 变了：新建了一个张量，Y 指向它
Y += X             # id 没变：原地改，复用原内存
Y[:] = Y + X       # id 也没变：先算临时结果，再拷回原位置
```

**我的判断：`Y += X` 和 `Y[:] = Y + X` 的区别在自动微分上。**
前者是原地操作，如果 `Y` 是某个需要求导的中间结果，autograd 可能报
「a leaf Variable that requires grad is being used in an in-place operation」，
因为它记下的旧值被改了，反向传播时梯度算错。
后者先算新张量再拷贝，计算图里记录的中间值是独立的，安全。
第 3 章手写 SGD 更新参数时，我要用 `param.data -= ...` 或者包在 `torch.no_grad()` 里，
就是为了绕开这个限制。

### 按轴运算

`X.sum(axis=0)` 消掉第 0 维（对列求和），`X.sum(axis=1)` 消掉第 1 维（对行求和）。
`keepdim=True` 保留成长度 1 的维度，这样可以直接和原张量广播相除——
做 softmax 减最大值时就是靠这个：

$$y_i = \frac{\exp(x_i - \max_j x_j)}{\sum_k \exp(x_k - \max_j x_j)}$$

## 2.2 数据预处理

流程固定四步：读表 → 处理缺失值 → 类别特征编码 → 转张量。

```python
import pandas as pd
import torch

# 用四行小表代替真实数据，坑一样能复现：Area 是数值列，Alley 是字符串列
data = pd.DataFrame({
    "Area": [8450.0, 9600.0, None, 11250.0],
    "Alley": ["Pave", None, "Grvl", None],
})
numeric = data.select_dtypes(include="number").columns
data[numeric] = data[numeric].fillna(data[numeric].mean())
data = pd.get_dummies(data, dummy_na=True)
X = torch.tensor(data.values, dtype=torch.float32)
```

### 踩坑记录

1. **`fillna(data.mean())` 在第 2 列是字符串时直接报错或静默出 NaN。**
   我第一版对整表 fillna，因为表里有一列 `Alley` 是字符串，pandas 把整列当 object，
   `mean()` 返回 Series 里没有这一列，结果那一列还是 NaN，转张量时报
   `can't convert np.ndarray of type numpy.object_`。正确做法是先分 numeric / 类别两路。
2. **`torch.tensor(df.values)` 不指定 dtype，整数列会让整个张量变成 int64。**
   后面除以均值就变成整数除法，全 0。必须显式 `dtype=torch.float32`。
3. **`df.values` 会丢掉列名顺序信息**，用了 `get_dummies` 之后列顺序是原列 + 各类别，
   训练和推理必须用同一份列顺序，否则特征对不上。稳妥做法是先 `reindex(columns=...)`。

`pd.get_dummies(..., dummy_na=True)` 会给缺失单独开一列 `Alley_nan`，
这是有意的：缺失本身可能就是信息（比如「没记录巷子类型」说明是高价房）。
