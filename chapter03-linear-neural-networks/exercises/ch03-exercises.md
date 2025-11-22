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
