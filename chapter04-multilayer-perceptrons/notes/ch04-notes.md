# 第 4 章 多层感知机 · 读书笔记

## 这一章的主线

```text
线性模型不够用（XOR 分不开）
   ↓ 加隐藏层
但隐藏层必须是「线性 + 非线性」的组合，否则等于没加   ← 4.1
   ↓ 于是模型有容量了
容量一大就过拟合（训练误差低、泛化误差高）             ← 4.4
   ↓ 两条路压容量
限制参数大小（权重衰减） / 随机丢神经元（dropout）     ← 4.5 / 4.6
   ↓ 压的时候发现
层数一多梯度会消失或爆炸，初始化方式很关键             ← 4.7
```

## 框架树

```text
多层感知机
├── 4.1 为什么需要非线性
│   ├── 仿射变换的复合仍是仿射变换（要证明）
│   ├── 激活函数：ReLU / sigmoid / tanh
│   └── 通用近似定理（知道结论即可）
├── 4.2 从零实现
│   ├── 参数：W1 (d,h), b1 (h), W2 (h,k), b2 (k)
│   ├── 前向：H = relu(X W1 + b1); O = H W2 + b2
│   ├── 损失：交叉熵（吃 logits）
│   └── 更新：手写 SGD，梯度已由 autograd 给出
├── 4.3 简洁实现
│   ├── nn.Sequential 按顺序前向
│   ├── net.apply(init_weights) 递归初始化
│   └── 训练循环和线性模型完全一样 ★
├── 4.4 模型选择
│   ├── 训练误差 / 泛化误差
│   ├── 模型容量、VC 维的直觉
│   └── 数据量、模型大小、正则强度三者的权衡
├── 4.5 权重衰减
│   ├── L2 正则 → 更新式里的 (1 - eta*lambda)
│   ├── 只衰减 weight，不衰减 bias
│   └── 为什么它能限制容量
├── 4.6 dropout
│   ├── 训练期以 p 概率置零
│   ├── 推理期用期望（inverted dropout 是训练期除以 1-p）
│   └── 只在训练时开，eval 必须关 ★
└── 4.7 数值稳定性
    ├── 梯度是连乘，深了就消失/爆炸
    ├── Xavier：sigma^2 = 2/(n_in + n_out)
    └── 与激活函数配套（ReLU 用 Kaiming）
```

标 ★ 的两处是这一章最容易写错、也是我实际写错的地方：
「训练循环和线性模型一样」意味着可以整套复用，我一开始又重写了一遍；
「eval 必须关 dropout」我漏了，导致验证指标比训练指标差一大截。

## 我的读法

4.1 和 4.4 是概念，读一遍 + 自己举反例；
4.2、4.5、4.6 必须动手，因为「为什么这么写」只有写错过才知道；
4.7 只求能解释现象，不追求推完整的初始化理论。

## 4.1 隐藏层与激活函数

### 两层网络的定义

$$\mathbf{H} = \sigma(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1), \qquad
\mathbf{O} = \mathbf{H}\mathbf{W}_2 + \mathbf{b}_2$$

$\mathbf{X} \in \mathbb{R}^{n\times d}$，$\mathbf{W}_1 \in \mathbb{R}^{d\times h}$，
$\mathbf{W}_2 \in \mathbb{R}^{h\times q}$。若 $\sigma$ 是恒等映射，则

$$\mathbf{O} = (\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
= \mathbf{X}\underbrace{(\mathbf{W}_1\mathbf{W}_2)}_{\mathbf{W}'} + \underbrace{(\mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2)}_{\mathbf{b}'}$$

也就是说，**两层仿射变换的复合等价于一层**，参数量从 $dh + hq$ 退化成 $dq$，
表达能力反而下降。所以激活函数不是「加个点缀」，它是隐藏层存在的唯一理由。

### 三个激活函数

$$\mathrm{ReLU}(x) = \max(x, 0), \qquad
\mathrm{sigmoid}(x) = \frac{1}{1+e^{-x}}, \qquad
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

导数：

$$\mathrm{ReLU}'(x) = \mathbb{1}[x > 0], \qquad
\mathrm{sigmoid}'(x) = \mathrm{sigmoid}(x)\left(1 - \mathrm{sigmoid}(x)\right) \le \frac{1}{4}$$

**我为什么选 ReLU 做默认：** sigmoid 的导数最大只有 0.25，多层连乘之后梯度按
$0.25^L$ 衰减，10 层就是 $10^{-6}$；ReLU 在正半轴的导数恒为 1，梯度不会因为层数衰减。
代价是负半轴梯度为 0（「神经元死亡」），实践里靠小学习率和合理初始化避开。

ReLU 在 $x=0$ 处不可导，PyTorch 里取次梯度 0。理论上不严谨，实践上完全没问题——
随机初始化下恰好落在 0 的概率是 0。

### 通用近似定理

含一个隐藏层、足够多神经元的网络可以以任意精度逼近紧集上的连续函数。
**我的判断：这个定理不该被过度解读。** 它只说「存在」，没说「梯度下降能找到」，
也没说「需要多少神经元」。所以「理论上一个隐藏层就够」这句话对实践没有指导意义，
真正决定深度的是「表示效率」：深网络用指数少的参数表达某些函数。

## 4.2 从零实现：参数量和更新顺序

### 参数量

$d=2, h=16, q=2$：

| 参数 | 形状 | 个数 |
|---|---|---|
| $\mathbf{W}_1$ | $2\times16$ | 32 |
| $\mathbf{b}_1$ | $16$ | 16 |
| $\mathbf{W}_2$ | $16\times2$ | 32 |
| $\mathbf{b}_2$ | $2$ | 2 |
| 合计 | | **82** |

一般公式：$dh + h + hq + q$。第 1 章的线性模型在这个任务上只有 $2\times2+2 = 6$ 个参数，
换成 MLP 是 82 个，13 倍——这就是「容量」的具体含义。

### 手写 SGD 的顺序

```python
loss = F.cross_entropy(mlp(Xb, params), yb)   # 1. 前向 + 损失
loss.backward()                                # 2. 反传，梯度累加到 .grad
with torch.no_grad():
    for p in params:
        p -= lr * p.grad                       # 3. 更新
        p.grad.zero_()                         # 4. 清零
```

**注意这里没有除以 `batch_size`。** 因为 `F.cross_entropy` 默认 `reduction='mean'`，
梯度已经是平均值；而第 3 章从零写线性回归时用的是 `.sum()`，所以要自己除。
两者必须恰好除一次，这是我从第 3 章带过来的教训，写在注释里防止以后再错。

### 参数用 list 管

从零版把四个参数放进一个 list，好处是更新和清零都是一个 for 循环；
坏处是没有名字，调试时只能靠下标。第 5 章会用 `nn.Parameter` 和 `named_parameters()`
解决这个问题——那一章就是专门讲「参数怎么管理」的。
