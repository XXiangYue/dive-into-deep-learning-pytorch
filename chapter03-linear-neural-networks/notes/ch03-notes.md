# 第 3 章 线性神经网络 · 读书笔记

## 这一章在讲什么

一句话：**把「模型 + 损失 + 优化」这三件东西接成一条能跑的流水线，并用最简单的模型
（线性 / softmax）把这条流水线跑通。** 后面加卷积、加残差，改的都是模型那一块，
流水线的骨架不变。

## 框架树

```text
线性神经网络
├── 3.1 线性回归
│   ├── 模型：y = w^T x + b
│   ├── 损失：平方损失 1/2 (y_hat - y)^2
│   ├── 解析解：w* = (X^T X)^{-1} X^T y
│   └── 优化：小批量随机梯度下降
├── 3.2 从零实现
│   ├── synthetic_data：造带噪声的线性数据
│   ├── data_iter：随机打乱 + 切批
│   ├── sgd：手动 param -= lr * grad / batch_size
│   └── 训练循环：前向 → 反传 → 更新 → 记录
├── 3.3 简洁实现
│   ├── nn.Sequential(nn.Linear(2,1))
│   ├── TensorDataset + DataLoader
│   ├── nn.MSELoss（默认 mean）
│   └── torch.optim.SGD
├── 3.4 softmax 回归
│   ├── 从回归到分类：输出变成 k 维
│   ├── one-hot 编码
│   ├── softmax：把 logits 变成概率
│   └── 交叉熵：-log(y_hat[y])
├── 3.5 图像分类数据集
│   └── 标签是类别下标，不是 one-hot（省内存）
├── 3.6 从零实现 softmax
│   ├── 按行归一化要 keepdim
│   ├── 交叉熵与数值稳定性（减最大值）
│   └── 准确率：argmax == y 的比例
└── 3.7 简洁实现
    └── CrossEntropyLoss 内部带 softmax，喂 logits
```

## 我的读法

- 3.1 的推导自己写一遍，尤其是「平方损失求导后那个 2 去哪了」；
- 3.2 严格不看书写训练循环，卡住的地方就是没懂的；
- 3.4 到 3.6 连着读，softmax 和交叉熵必须一起理解，拆开看都是玄学；
- 3.7 只当 API 文档，但要验证「它内部已经做了 softmax」这件事。

## 3.1 线性回归

### 模型

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

写成矩阵形式（一次算一个批量的 $n$ 个样本）：$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$，
其中 $\mathbf{X} \in \mathbb{R}^{n \times d}$，$\mathbf{w} \in \mathbb{R}^{d}$。
注意 PyTorch 里 `nn.Linear(d, 1)` 的权重形状是 `(1, d)`，输出是 `(n, 1)`；
我手写的时候用 `(d, 1)` 更顺手，两种写法都对，但要在一个文件里统一，别混。

### 损失

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2}\left(\hat{y}^{(i)} - y^{(i)}\right)^2, \qquad
L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n} l^{(i)}(\mathbf{w}, b)$$

### 对 $\mathbf{w}$ 的梯度

令残差 $r^{(i)} = \hat{y}^{(i)} - y^{(i)}$，则

$$\frac{\partial l^{(i)}}{\partial \mathbf{w}} = r^{(i)}\,\mathbf{x}^{(i)}, \qquad
\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n}\sum_{i=1}^{n} r^{(i)}\,\mathbf{x}^{(i)} = \frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y})$$

$$\frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^{n} r^{(i)}$$

**我为什么这么理解梯度：** $\mathbf{X}^\top \mathbf{r}$ 的意思是「每个特征拿自己的取值去
乘上它造成的误差，再加起来」。哪个特征和误差的相关性大，它分到的梯度就大。
这和第 2 章矩阵乘的列视角是一回事——$\mathbf{X}^\top\mathbf{r}$ 就是把 $\mathbf{r}$ 当成
权重，对各列（各特征）做加权求和。

### 一个具体的数

$\mathbf{w}=[2, -3.4]$，$b=4.2$，噪声标准差 $0.01$，1000 个样本。
损失理论下界是噪声方差的一半：$\frac{1}{2}\sigma^2 = 0.00005$。
所以训练到 loss ≈ 5e-5 就到头了，再往下就是在拟合噪声。这个数我要记住，
不然看到 loss 卡在 5e-5 会以为没收敛。

### 解析解

把偏置并进 $\mathbf{w}$（在 $\mathbf{X}$ 末尾加一列全 1），损失对 $\mathbf{w}$ 求导置零：

$$\nabla_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 = 2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) = \mathbf{0}$$

$$\Rightarrow\quad \mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}
\quad\Rightarrow\quad \mathbf{w}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

### 那为什么还要 SGD

三个理由，按我自己的理解排序：

1. **$\mathbf{X}^\top\mathbf{X}$ 是 $d \times d$ 的矩阵，求逆是 $O(d^3)$。**
   深度学习里 $d$ 动辄上百万，这个代价不可接受。而 SGD 每步只要 $O(bd)$（$b$ 是批量）。
2. **解析解只对平方损失成立。** 换成交叉熵、加上正则项、加上非线性层，就没有闭式解了。
3. **数据可能装不进内存。** 解析解要看到全部 $\mathbf{X}$；SGD 一个批量就能更新一次。

**我的判断：解析解的价值在于它是一个「可以对齐的基准」。**
我先用解析解算出 $\mathbf{w}^*$，再跑 SGD，两者差多少就知道 SGD 实现对不对。
`code/01` 里最后一步就是拿解析解对照的。

### 小批量 SGD

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \nabla_{\mathbf{w}} l^{(i)}(\mathbf{w}, b)$$

为什么除以 $|\mathcal{B}|$：让更新量的尺度和批量大小无关。
不除的话，批量从 10 改到 100，有效学习率会涨 10 倍，这是调 batch size 时最常见的隐形坑。

**梯度的无偏性**：$\mathbb{E}_{\mathcal{B}}[\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\nabla l^{(i)}] = \nabla L$，
所以小批量梯度是真实梯度的无偏估计，方差随批量增大而减小（约 $1/|\mathcal{B}|$）。

## 3.2 从零实现：两个真实的坑

### 坑一：`data_iter` 不打乱，loss 会台阶式下降

第一版我忘了 `random.shuffle(indices)`，结果每个 epoch 都按同一顺序喂数据。
现象很怪：loss 在 epoch 内部跳动很小，跨 epoch 才有明显下降，而且对学习率极其敏感。
原因是同一批样本总是同时出现，梯度方向高度相关，等于把批量变小了。
打乱之后 loss 曲线立刻平滑。

### 坑二：`param -= lr * param.grad` 直接报错

```text
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

`param` 是叶子节点且 `requires_grad=True`，`-=` 是原地操作，autograd 不许这么改它
（它要留着旧值算梯度）。三种写法，我最后选了第一种：

```python
with torch.no_grad():          # 官方推荐：显式声明这段不建图
    param -= lr * param.grad / batch_size
    param.grad.zero_()

param.data -= lr * param.grad / batch_size   # 也能跑，但绕过了 autograd 的检查
param = param - lr * param.grad / batch_size # 错：只是把局部名字重新绑定，原参数没变
```

第三种最容易写错，而且不报错——训练照跑，参数就是不动。我第一次就是这么写的，
跑了 5 个 epoch 发现 loss 一直是 4.2（正好是初始值附近），查了两小时。

### 顺序也很重要

`param.grad.zero_()` 必须在更新**之后**。如果先清零再更新，那一步的梯度就是 None / 0，
参数永远不动。更稳妥的写法是在循环开头清零，和 `optimizer.zero_grad()` 的位置一致。

## 3.3 softmax 回归

### 从回归到分类

分类的输出是 $k$ 个类别的置信度（logits）$\mathbf{o} \in \mathbb{R}^k$。
要把它变成概率分布，需要两个条件：非负、和为 1。

$$\hat{y}_j = \mathrm{softmax}(\mathbf{o})_j = \frac{\exp(o_j)}{\sum_{l=1}^{k}\exp(o_l)}$$

指数保证了非负，除以总和保证了归一化。而且它保序：$o_j$ 最大的那个类别概率也最大，
所以预测类别可以直接 `argmax(o)`，不必先算 softmax——这是 3.7 简洁实现里
`CrossEntropyLoss` 吃 logits 的原因。

### one-hot 标签

真实标签 $y$（一个整数下标）写成 one-hot 向量 $\mathbf{y}$，只有 $y$ 位置是 1。

$$\mathbf{y} = [0, \dots, 1, \dots, 0]$$

**为什么要 one-hot：** 因为交叉熵的定义是 $-\sum_j y_j \log \hat{y}_j$，
用 one-hot 代进去只剩一项 $-\log \hat{y}_y$。如果直接用整数下标，
$-\log \hat{y}_y$ 这个写法和 one-hot 版本完全等价，所以实现里存下标、计算时按索引取，
省内存也省一次 one-hot 乘法。

### 交叉熵

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^{k} y_j \log \hat{y}_j = -\log \hat{y}_y$$

代入 softmax：

$$l = -\log\frac{\exp(o_y)}{\sum_j \exp(o_j)} = -o_y + \log\sum_j \exp(o_j)$$

**这个形式很关键：** 第一项 $-o_y$ 是「正确类别的 logit 越小损失越大」，
第二项 $\log\sum_j e^{o_j}$ 是 log-sum-exp，起到「压制其他类别」的作用。
两者合起来正好是我做对时 $l \to 0$、做错时 $l$ 很大。

### 为什么不用平方损失做分类

平方损失对 logits 的梯度会带一个 $\hat{y}_j(1 - \hat{y}_j)$ 的因子，
当预测错得很离谱（$\hat{y}_y \approx 0$）时这个因子趋于 0，梯度消失，学不动。
交叉熵的梯度没有这个问题（见下一条笔记）。这也是「损失函数要匹配输出分布」的例子。

### 交叉熵对 logits 的梯度（这一章最重要的推导）

记 $S = \sum_{l} \exp(o_l)$，则 $\hat{y}_j = \exp(o_j)/S$。

先算 softmax 自己的雅可比：

$$\frac{\partial \hat{y}_j}{\partial o_i} = \frac{\exp(o_j)\cdot \mathbb{1}[i=j]\cdot S - \exp(o_j)\exp(o_i)}{S^2}
= \hat{y}_j\left(\mathbb{1}[i=j] - \hat{y}_i\right)$$

再对 $l = -\sum_j y_j \log \hat{y}_j$ 求导（$y$ 是 one-hot，$\sum_j y_j = 1$）：

$$\frac{\partial l}{\partial o_i} = -\sum_j \frac{y_j}{\hat{y}_j}\cdot\frac{\partial \hat{y}_j}{\partial o_i}
= -\sum_j \frac{y_j}{\hat{y}_j}\hat{y}_j\left(\mathbb{1}[i=j] - \hat{y}_i\right)
= -\left(y_i - \hat{y}_i\sum_j y_j\right)$$

$$\boxed{\frac{\partial l}{\partial o_i} = \hat{y}_i - y_i}$$

**我为什么觉得这个结果漂亮：** 它和平方损失的梯度 $\hat{y} - y$ 形式完全一样，
但这里的 $\hat{y}$ 是概率、没有那个 $p(1-p)$ 的衰减因子。所以预测错得越离谱，
梯度越大，学得越快。这就是「交叉熵配 softmax」的标准答案。

一个批量上的平均损失梯度是 $\frac{1}{n}\sum_i (\hat{\mathbf{y}}^{(i)} - \mathbf{y}^{(i)})$，
在 torch 里就是 `(softmax(o) - onehot(y)).mean(dim=0)` —— 但实际不需要我手算，
`nn.CrossEntropyLoss` 一次搞定。

### 数值坑（这一章我在这里返工）

$\exp(o)$ 在 $o \approx 90$ 时就超出 float32 上限（$3.4\times10^{38}$，$\ln$ 之后约 88.7）。
logits 一大，softmax 直接输出 nan，loss 也是 nan。标准做法是先减去每行最大值：

$$\hat{y}_j = \frac{\exp(o_j - \max_l o_l)}{\sum_l \exp(o_l - \max_l o_l)}$$

数学上完全等价（分子分母同乘 $e^{-\max}$），但指数最大是 $\exp(0)=1$，不会溢出。
交叉熵那边同理，要用 log-sum-exp 形式，不能先 softmax 再取 log。

## 3.4 / 3.5 数据集：我为什么不用 Fashion-MNIST

原书 3.5 节用框架自带的数据集接口下载 Fashion-MNIST，30MB，第一次运行要联网。
我换成合成的三类二维高斯团：

```python
CENTERS = [[2.0, 2.0], [-2.0, 1.5], [0.0, -2.5]]
X_k = torch.normal(CENTERS[k], std=0.7, size=(300, 2))
```

### 三个理由

1. **可离线、可秒跑。** 900 个样本 × 2 维，CPU 上 10 个 epoch 一秒内跑完，
   调超参不用等。
2. **重点是机制不是图像。** softmax、one-hot、交叉熵、准确率这四件事跟输入是图像
   还是二维点毫无关系。我在二维上还能把决策边界画出来（第 3.6 节末尾）。
3. **可以控制难度。** 改 `std` 就能从「完全可分」调到「重叠严重」，
   观察 acc 从 0.99 掉到 0.7 的过程，比在真实数据集上猜为什么掉更有用。

### 和真实数据集对齐的地方

- 标签用**整数下标** `torch.long`，形状 `(n,)`，不是 one-hot；
- 训练集 / 验证集手动切 800 / 100，验证集不参与梯度；
- 特征做标准化（减均值除标准差），因为 softmax 的 logits 对特征尺度敏感；
- `DataLoader` 的用法和真实数据集完全一致，换成 Fashion-MNIST 只要改数据源。

### 一个必须注意的点

`torch.normal(mean_tensor, std, size)` 的签名和 `torch.normal(mean, std, size)` 不一样：
传张量做均值时不能再传 `size`。我第一版写了
`torch.normal(center, 0.7, (300, 2))` 直接 TypeError，正确写法是把 $n$ 放进 size 或者
用 `center.expand(300, 2)` 先铺开。

## 3.6 / 3.7 简洁实现对照

| 环节 | 从零版 | 简洁版 |
|---|---|---|
| 参数 | 手动 `torch.normal(..., requires_grad=True)` | `nn.Linear(2, 1)` 自动建参数 |
| 前向 | `X @ w + b` | `net(X)` |
| 损失 | `((y_hat - y)**2 / 2).mean()` | `nn.MSELoss()`（默认 mean） |
| 更新 | `param -= lr * grad / batch_size` | `trainer.step()` |
| 清零 | `param.grad.zero_()` | `trainer.zero_grad()` |
| 批量 | 手写 `data_iter` | `DataLoader(TensorDataset(...))` |

三个容易搞错的差别：

1. **`MSELoss` 默认已经除过 batch size**（`reduction='mean'`）。
   所以简洁版的 `l.backward()` 之后不用再除 `batch_size`，再除一次等于学习率缩小 10 倍。
   我第一次照抄从零版的写法，`lr=0.03` 跑了 20 个 epoch 都没收敛，就是这个原因。
2. **`nn.Linear(2, 1)` 的权重形状是 `(1, 2)`，不是 `(2, 1)`**。
   对照真值时要 `net[0].weight.data.flatten()`，而不是直接比。
3. **`CrossEntropyLoss` 内部已经做了 log-softmax**，喂给它的是 logits 不是概率。
   我第一版把 `softmax(net(X))` 喂进去，loss 一直在 0.6 附近下不去——
   等于做了两次 softmax，概率被压得更平。

### 关于「从零版还有没有意义」

有。写完从零版我才真正理解 `optimizer.step()` 里那个 `/batch_size` 是从哪来的，
也才知道 `zero_grad()` 是必须的而不是可选的。第 4 章之后我全都用简洁版，
但每章的从零版还是照写，因为要改网络结构的时候，知道自己改的是哪一行。
