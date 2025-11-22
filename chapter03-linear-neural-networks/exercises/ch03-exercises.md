# 第 3 章 习题与解答

## 习题 3.1 证明平方损失对 $\mathbf{w}$ 的梯度是 $\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y})/n$

$$L = \frac{1}{2n}\sum_{i=1}^{n}\left(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}\right)^2$$

记 $r^{(i)} = \mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}$，对第 $j$ 个分量求偏导：

$$\frac{\partial L}{\partial w_j} = \frac{1}{n}\sum_{i=1}^{n} r^{(i)}\cdot x_j^{(i)}$$

写成向量形式就是 $\frac{1}{n}\mathbf{X}^\top\mathbf{r}$，而 $\mathbf{r} = \hat{\mathbf{y}} - \mathbf{y}$。
$\square$

**注意我在实现里的一个不一致：** `squared_loss` 返回的是逐样本损失向量，
`train` 里先 `.sum()` 再 `.backward()`，所以拿到的梯度是 $\mathbf{X}^\top\mathbf{r}$（没除 $n$）；
`sgd` 里再除 `batch_size`，合起来才是 $\frac{1}{|\mathcal{B}|}\mathbf{X}^\top\mathbf{r}$。
两处除法必须恰好一次，多除一次学习率就白调了。

## 习题 3.2 从零实现里的 `data_iter` 为什么每个 epoch 都要打乱

不打乱时，同一个批量的样本组成固定，梯度方向是固定几个样本的梯度的平均。
整个 epoch 的更新方向高度相关，等价于把有效批量变小、方差变大。
实践现象是 loss 在 epoch 内几乎不降、跨 epoch 才跳一下，对学习率特别敏感。
打乱之后每个批量的组成是随机的，梯度是真实梯度的无偏估计。

顺带一个细节：`random.shuffle` 用的是 Python 的随机源，
所以要复现结果必须同时 `random.seed` 和 `torch.manual_seed`，
只固定 torch 那个是不够的——这是我第 1 章自检脚本返工的原因。

## 习题 3.3 解释「小批量梯度的期望等于全量梯度」

设训练集有 $n$ 个样本，均匀无放回地抽一个大小 $|\mathcal{B}|$ 的批量。则

$$\mathbb{E}\left[\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\nabla l^{(i)}\right]
= \frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\mathbb{E}\left[\nabla l^{(i)}\right]
= \frac{1}{n}\sum_{i=1}^{n}\nabla l^{(i)} = \nabla L$$

第二个等号用到「每个样本被抽中的概率是 $|\mathcal{B}|/n$」。
所以小批量梯度是无偏的，只是方差不为零；方差约为 $\sigma^2/|\mathcal{B}|$
（$\sigma^2$ 是单样本梯度的方差），这是「批量越大越稳」的定量版本。

## 习题 3.4 推导 $\partial l/\partial o_i = \hat{y}_i - y_i$

这一题在笔记 3.3 里已经完整推过，这里只写结论和最直观的解释：

$$\frac{\partial l}{\partial o_i} = \hat{y}_i - y_i$$

意思是「预测概率减真实 one-hot」。正确类别上 $y_i=1$，梯度是 $\hat{y}_i - 1 < 0$，
沿负梯度走就等价于**把这个 logit 往上推**；错误类别上 $y_i = 0$，梯度是 $\hat{y}_i > 0$，
更新就是**把它的 logit 往下压**。所以一次更新同时在做「抬高正确类、压低错误类」。

而且梯度的绝对值正好是「错误程度」：$\hat{y}_i$ 越接近 1（越自信地错），梯度越大。
这解释了为什么交叉熵比平方损失在分类上收敛快。

## 习题 3.5 $k=2$ 时 softmax 回归是否退化为 logistic 回归

是。取 $o_1 = \mathbf{w}_1^\top\mathbf{x} + b_1$，$o_2 = \mathbf{w}_2^\top\mathbf{x} + b_2$，则

$$\hat{y}_1 = \frac{e^{o_1}}{e^{o_1} + e^{o_2}} = \frac{1}{1 + e^{-(o_1 - o_2)}}
= \sigma\left((\mathbf{w}_1 - \mathbf{w}_2)^\top\mathbf{x} + (b_1 - b_2)\right)$$

正是 sigmoid。所以二分类用 logistic 和用 2 类 softmax 在数学上等价，
区别只在参数化（softmax 有两组参数，但差值才是有效的，存在冗余）。
**冗余的后果：** 参数不唯一，L2 正则会把这组冗余「分配」掉，属于良性冗余。

## 习题 3.6（没做完）比较解析解与 SGD 解在「特征共线」时的差异

想法：造两个高度相关的特征（$x_2 \approx x_1 + \epsilon$），看正规方程的解是否爆炸、
SGD 的解是否稳定。

做到一半卡住了：`torch.linalg.solve` 在 $\mathbf{X}^\top\mathbf{X}$ 接近奇异时
给我的解已经到 $10^4$ 量级，SGD 却还停在初始值附近——我不确定这是 SGD 收敛慢
还是它其实收敛到了另一个「好」的解。查了资料说这涉及最小范数解，属于第 11 章
优化算法和第 4 章权重衰减的内容，先记着，等读完再回来。

至少确认了一件事：**特征共线时解析解不可信，梯度类方法更稳。** 这一条已经够用了。
