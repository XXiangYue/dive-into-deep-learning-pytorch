# 第 4 章 习题与解答

## 习题 4.1 证明：若激活函数是恒等映射，两层网络退化为单层

$$\mathbf{O} = (\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
= \mathbf{X}(\mathbf{W}_1\mathbf{W}_2) + (\mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2)$$

令 $\mathbf{W}' = \mathbf{W}_1\mathbf{W}_2$（形状 $d\times q$），
$\mathbf{b}' = \mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2$，则 $\mathbf{O} = \mathbf{X}\mathbf{W}' + \mathbf{b}'$，
正是单层线性模型。

**参数量对比：** 两层是 $dh + h + hq + q$，单层是 $dq + q$。
$d=q=2, h=16$ 时是 82 vs 6。所以加了隐藏层但去掉非线性，参数量涨 13 倍，
表达能力却没变——相当于把同样的函数用更高的维数重写了一遍。

## 习题 4.2 算 MLP 的参数量（$d=2, h=16, q=2$）

| 参数 | 形状 | 个数 |
|---|---|---|
| $\mathbf{W}_1$ | $2\times16$ | 32 |
| $\mathbf{b}_1$ | 16 | 16 |
| $\mathbf{W}_2$ | $16\times2$ | 32 |
| $\mathbf{b}_2$ | 2 | 2 |
| 合计 | | 82 |

程序里 `sum(p.numel() for p in params)` 输出 82，和手算一致。
一般公式：$dh + h + hq + q = h(d + q + 1) + q$。

## 习题 4.3 ReLU 在 0 处不可导，为什么实践上没问题

$$\mathrm{ReLU}'(x) = \begin{cases} 1, & x > 0\\ 0, & x < 0\end{cases}$$

$x=0$ 处左导 0、右导 1，不可导。但：

1. 随机初始化 + 连续分布的数据，某个预激活恰好等于 0 的概率为 0；
2. 框架用次梯度（PyTorch 取 0），在最优点附近不影响收敛方向；
3. 真正的问题不是 0 点，而是负半轴梯度恒为 0 导致「神经元死亡」。
   lr 太大时一批神经元永久失活，这才是实践中的坑。

对照：sigmoid/tanh 处处可导但导数有上界（sigmoid 最大 0.25），
深网络里连乘会消失。所以「可导」不是选激活函数的唯一标准，
**导数会不会消失才是**。

## 习题 4.4 推导带 L2 正则的 SGD 更新式

$$L_{reg} = L + \frac{\lambda}{2}\|\mathbf{w}\|_2^2, \qquad
\nabla_{\mathbf{w}} L_{reg} = \nabla_{\mathbf{w}} L + \lambda\mathbf{w}$$

$$\mathbf{w} \leftarrow \mathbf{w} - \eta(\mathbf{g} + \lambda\mathbf{w})
= (1 - \eta\lambda)\mathbf{w} - \eta\mathbf{g}$$

所以每一步先把自己的权重乘上 $(1-\eta\lambda)$，再走梯度步。

**稳定性条件：** 只看衰减部分，$\mathbf{w}_{t+1} = (1-\eta\lambda)\mathbf{w}_t$，
要收敛必须 $|1-\eta\lambda| < 1$，即 $0 < \eta\lambda < 2$。
超过 2 权重会震荡放大。这就是为什么 $\lambda$ 必须配着 $\eta$ 调——
$\eta$ 增大 10 倍，等效的 $\lambda$ 也放大了 10 倍。

## 习题 4.5 dropout 的期望与推理缩放

设某单元激活为 $h$，训练时以概率 $p$ 置零。

**不缩放版本：** 训练期期望 $\mathbb{E}[h'] = (1-p)h$，推理期输出 $h$。
不一致，推理时每层被放大 $1/(1-p)$ 倍。

**inverted 版本：** 训练期输出 $h' = h\cdot m/(1-p)$，
其中 $m\sim\mathrm{Bernoulli}(1-p)$，则

$$\mathbb{E}[h'] = \frac{(1-p)h}{1-p} = h$$

与推理期一致。方差是

$$\mathrm{Var}[h'] = \frac{h^2}{(1-p)^2}\cdot p(1-p) = \frac{p}{1-p}h^2$$

$p \to 1$ 时方差爆炸，这是 dropout 不能设太大的原因之一（我实测 p=0.5 还行，p=0.8 直接训不动）。

## 习题 4.6（半途）dropout 为什么等价于集成

想证明：对 $n$ 个隐藏单元，dropout 相当于从 $2^n$ 个子网络中采样训练，
推理时用全部单元相当于对这些子网络做平均。

已经想清楚的部分：每个 mask $m \in \{0,1\}^n$ 定义了一个子网络 $f_m$，
训练时每步只更新 $f_m$ 的参数（其他参数梯度为 0），所以每个子网络都被训练过。
推理时 $\mathbb{E}_m[f_m(x)]$ 用一阶近似可以写成 $f(x)$（若 $f_m$ 对 $m$ 近似线性）。

没做出来的部分：这些子网络**共享参数**，所以不是独立集成，
标准的集成方差公式套不上。我查到有工作（Baldi & Sadowski）从「几何平均」角度给了解释，
但还没读明白，先记着。
