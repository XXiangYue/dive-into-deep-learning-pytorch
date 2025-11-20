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
