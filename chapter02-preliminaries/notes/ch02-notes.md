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

## 2.3 线性代数

### 矩阵乘的两套读法

设 $\mathbf{A} \in \mathbb{R}^{m \times n}$，$\mathbf{x} \in \mathbb{R}^{n}$。

**行视角**（每个输出是一个内积）：

$$(\mathbf{A}\mathbf{x})_i = \mathbf{a}_{i,:}^\top \mathbf{x} = \sum_{j=1}^{n} a_{ij} x_j$$

**列视角**（输出是列的线性组合）：

$$\mathbf{A}\mathbf{x} = \sum_{j=1}^{n} x_j\, \mathbf{a}_{:,j}$$

**我为什么两个都要记：** 行视角适合理解「一个神经元就是一个内积」，
所以线性层 `nn.Linear(in, out)` 的权重矩阵是 `(out, in)`，每行对应一个输出神经元。
列视角适合理解「通道混合」，1×1 卷积就是把 $C_{\text{in}}$ 个通道按权重组合成
$C_{\text{out}}$ 个通道，本质是一次矩阵乘。第 6 章会再用到。

### 范数

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i} x_i^2}, \qquad \|\mathbf{X}\|_F = \sqrt{\sum_{i,j} x_{ij}^2}$$

L2 范数对 $\mathbf{x}$ 的梯度是 $\mathbf{x} / \|\mathbf{x}\|_2$（当 $\mathbf{x} \ne \mathbf{0}$），
这个结果我在 2.5 节用 autograd 验证过。L1 范数在 0 处不可导，所以权重衰减一般用 L2。
权重衰减项加在损失里是 $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$，梯度是 $\lambda \mathbf{w}$，
系数 $\frac{1}{2}$ 又是为了消掉平方导出的 2。

### 按轴求和的形状

`X.sum(axis=0)` 结果是 `(n,)`，不是 `(1, n)`；`keepdim=True` 才是 `(1, n)`。
写 `X / X.sum(axis=1)` 时如果忘了 keepdim，广播会按尾部对齐，形状全乱——
这是我这次实际报错的地方，报错信息是 `The size of tensor a (3) must match the size of tensor b (2)`。

## 2.4 微积分

### 导数与梯度

一元导数：$\dfrac{df}{dx} = \lim_{h \to 0}\dfrac{f(x+h) - f(x)}{h}$。
多元函数对每个变量求偏导，拼成梯度向量：

$$\nabla_{\mathbf{x}} f = \left[\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right]^\top$$

### 链式法则

若 $y = f(u)$，$u = g(x)$，则

$$\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}$$

多元版本（反向传播的骨架）：设 $z = f(\mathbf{y})$，$\mathbf{y} = g(\mathbf{x})$，则

$$\frac{\partial z}{\partial x_i} = \sum_{j} \frac{\partial z}{\partial y_j}\frac{\partial y_j}{\partial x_i}$$

**我的判断：这个求和号就是「一个变量被多条路径用到时，梯度要累加」。**
残差连接 $y = f(x) + x$ 的梯度里天然多一个恒等项 $\partial x/\partial x = 1$，
所以深网络的梯度不会一路衰减到 0——第 7 章 ResNet 的全部道理就在这一行。

### 为什么沿负梯度走

一阶泰勒展开：$f(\mathbf{x} + \eta \mathbf{d}) \approx f(\mathbf{x}) + \eta \nabla f^\top \mathbf{d}$。
要让函数值下降，就要让 $\nabla f^\top \mathbf{d} < 0$；在 $\|\mathbf{d}\|=1$ 的约束下，
$\mathbf{d} = -\nabla f / \|\nabla f\|$ 使内积最小（柯西–施瓦茨），即下降最快。
所以梯度下降是

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$$

**为什么不直接解 $\nabla f = 0$：** 线性回归可以（正规方程），
但深层网络的参数上百万且非凸，解方程既算不动也没有唯一解。这是优化算法存在的理由。

## 2.5 自动微分

### 它到底做了什么

前向时记录一张计算图：每个张量记住「我是由谁、用什么算子算出来的」。
反向时从输出往输入走，每个节点只需要把自己的局部雅可比乘上上游传来的梯度。

$$\frac{\partial L}{\partial \mathbf{x}} = \sum_{\text{path}} \prod_{\text{edge}} J_{\text{edge}}$$

和符号求导的区别：符号求导会展开成巨大的表达式（$f$ 复用时指数爆炸），
自动微分的复杂度是 $O(\text{图的大小})$，与求几次导无关。

### 三个必须记住的接口细节

1. **梯度是累加的**：`backward()` 不清零，`.grad` 会叠加。每轮必须 `zero_()`。
2. **`backward()` 只能对标量调用**（除非给 `gradient` 参数）。对向量 `y` 调 `y.backward()`
   会报 `grad can be implicitly created only for scalar outputs`。
   想要向量-雅可比积就传 `y.backward(torch.ones_like(y))`。
3. **`detach()` 断图**：`u = y.detach()` 之后 $u$ 不再追踪历史，
   后面所有基于 $u$ 的计算都不会把梯度传回 $y$。d2l 里更新参数就是用这个思路。

### 二阶导

要算 $\partial^2 y/\partial x^2$，第一次 `backward` 时必须 `create_graph=True`，
否则计算图已经被释放：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3
dy = torch.autograd.grad(y, x, create_graph=True)[0]   # 3x^2
d2y = torch.autograd.grad(dy, x)[0]                    # 6x
```

**我的判断：二阶导现在不用，但第 11 章讲牛顿法、第 4 章讲 Hessian 的时候会回来找这段。**

### 一个容易误解的点

叶子张量（用户直接创建的）的 `.grad` 会被保留；中间结果的 `.grad` 默认是 `None` 并会被释放。
想让中间张量保留梯度得 `x.retain_grad()`。我一开始以为所有 `.grad` 都能看，找了半天为什么是 None。

## 2.6 概率

### 基本关系

条件概率：

$$P(A \mid B) = \frac{P(A, B)}{P(B)}$$

贝叶斯公式：

$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

乘法法则 $P(A,B) = P(A\mid B)P(B)$，多个变量的链式分解
$P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})$。
**这个式子就是第 8 章语言模型的目标函数**，看着平平无奇，但它把「整段文本的概率」
拆成了每个位置一个条件概率，才能逐 token 训练。

### 期望、方差

$$\mathbb{E}[X] = \sum_{x} x\, p(x), \qquad \mathrm{Var}[X] = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

第二条等式我手推过：展开平方后交叉项是 $-2\mathbb{E}[X]\mathbb{E}[X]$，
剩下的正是 $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$。用这条算方差比第一条少一次遍历。

### 交叉熵 = 负对数似然

设真实标签是 one-hot 的 $\mathbf{y}$，预测分布 $\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})$：

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j} y_j \log \hat{y}_j = -\log \hat{y}_{y}$$

因为 $\mathbf{y}$ 只有一个位置是 1。所以交叉熵就是「正确类别的预测概率的负对数」，
也就是该样本在模型下的负对数似然。最小化交叉熵 = 最大化似然。

**我的判断：这个等式让「损失函数」和「概率模型」接上了。**
第 3 章只要记住一件事：softmax 输出被解释成概率，交叉熵就是它的负对数似然，
推梯度时 $-\log \mathrm{softmax}$ 里的归一化项会自动消掉。

### 抽样

`torch.multinomial(probs, num_samples, replacement=True)` 按概率抽样。
`torch.normal(mean, std, size)` 抽正态。经验分布随样本数收敛到真实分布（大数定律），
我在 `code/04_probability.py` 里用 100 / 1000 / 10000 三档看了误差怎么降。
