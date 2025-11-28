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
