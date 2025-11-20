"""线性回归的从零实现。

真值取 w = [2.0, -3.4]，b = 4.2，噪声标准差 0.01，1000 个样本。
全部手写：造数据、切批量、前向、平方损失、SGD、训练循环，不用 nn 也不用 optim。

用法：python code/01_linear_regression_scratch.py
"""

import random

import torch

# 真值。训练结束后要把估计值和它对比，误差应该在 1e-3 量级
TRUE_W = torch.tensor([2.0, -3.4])
TRUE_B = 4.2


def synthetic_data(w: torch.Tensor, b: float, num_examples: int, noise: float = 0.01):
    """生成 y = Xw + b + eps，返回 X (n, d) 和 y (n, 1)。

    噪声必须加，否则损失的最优值是 0，看不出「收敛到什么水平算正常」。
    """
    X = torch.normal(0.0, 1.0, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0.0, noise, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size: int, features: torch.Tensor, labels: torch.Tensor):
    """按批量产出 (X, y)。每个 epoch 先打乱，避免梯度方向总是一致。"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 最后一批可能不足 batch_size，用 min 截断
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """线性前向：X (n, d) @ w (d, 1) + b -> (n, 1)。"""
    return X @ w + b


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """平方损失，返回每个样本的损失向量（不求和也不平均）。

    除以 2 是为了让梯度里的系数消掉，对最优解没有影响。
    y.reshape(y_hat.shape) 是必须的：y 可能来自不同形状的切片。
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def analytic_solution(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """正规方程 w* = (X^T X)^{-1} X^T y，用来给 SGD 的结果做基准。

    先把偏置并进 X（末尾加一列 1），一次解出 [w, b]。
    """
    ones = torch.ones(features.shape[0], 1)
    X = torch.cat([features, ones], dim=1)
    return torch.linalg.solve(X.T @ X, X.T @ labels).flatten()
