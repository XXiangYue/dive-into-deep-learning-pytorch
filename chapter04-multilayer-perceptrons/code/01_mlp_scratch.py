"""多层感知机从零实现：2 -> 16 -> 2 的两层网络，二分类。

数据用 sklearn 的 make_moons（两个交错的半月），200 个点、噪声 0.2。
选它的理由：线性模型在它上面必然欠拟合（卡在 0.86 左右），
加一个隐藏层就能到 0.94 以上，非线性的作用一眼可见。

用法：python code/01_mlp_scratch.py
"""

import torch
from sklearn.datasets import make_moons
from torch import nn

NUM_INPUTS, NUM_HIDDENS, NUM_OUTPUTS = 2, 16, 2


def make_data(n_samples: int = 200, noise: float = 0.2, seed: int = 42):
    """造数据 + 按特征标准化。返回 X (n, 2) float32，y (n,) int64。"""
    X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    # 标准化：两个特征的量纲差一点点，学习率就得跟着调，不如先统一
    X = (X - X.mean(dim=0, keepdim=True)) / X.std(dim=0, keepdim=True)
    return X, y


def init_params(num_inputs: int = NUM_INPUTS, num_hiddens: int = NUM_HIDDENS,
                num_outputs: int = NUM_OUTPUTS, scale: float = None):
    """初始化两层 MLP 的四个参数，全部 requires_grad。

    scale=None 时用 Xavier：sigma = sqrt(2/(n_in + n_out))，
    作用是让每层输出的方差在层间大致保持稳定。
    传具体数值可以复现「初始化太大 → 梯度爆炸」的实验。
    """
    def rand(shape, fan_in, fan_out):
        std = scale if scale is not None else (2.0 / (fan_in + fan_out)) ** 0.5
        return (torch.randn(shape) * std).requires_grad_(True)

    W1 = rand((num_inputs, num_hiddens), num_inputs, num_hiddens)
    b1 = torch.zeros(num_hiddens, requires_grad=True)
    W2 = rand((num_hiddens, num_outputs), num_hiddens, num_outputs)
    b2 = torch.zeros(num_outputs, requires_grad=True)
    return [W1, b1, W2, b2]


def relu(X: torch.Tensor) -> torch.Tensor:
    """ReLU：负半轴取 0，正半轴原样。用 torch.max 实现，避免写 python if。"""
    return torch.max(X, torch.zeros_like(X))


def mlp(X: torch.Tensor, params: list) -> torch.Tensor:
    """前向：H = relu(X W1 + b1)，O = H W2 + b2。返回 logits。"""
    W1, b1, W2, b2 = params
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


def train(params: list, X: torch.Tensor, y: torch.Tensor,
          epochs: int = 30, lr: float = 0.1, batch_size: int = 20) -> list:
    """手写训练循环，返回每个 epoch 的 (loss, acc)。

    F.cross_entropy 默认 reduction='mean'，所以梯度已经是按批平均的，
    更新时不要再除以 batch_size（和第 3 章从零版的写法不同，那里用的是 sum）。
    """
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i: i + batch_size]
            loss = nn.functional.cross_entropy(mlp(X[idx], params), y[idx])
            loss.backward()
            with torch.no_grad():
                for p in params:
                    p -= lr * p.grad
                    p.grad.zero_()
        with torch.no_grad():
            logits = mlp(X, params)
            epoch_loss = float(nn.functional.cross_entropy(logits, y))
            acc = float((logits.argmax(dim=1) == y).float().mean())
        history.append((epoch_loss, acc))
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1:>2}  loss {epoch_loss:.4f}  acc {acc:.3f}")
    return history
