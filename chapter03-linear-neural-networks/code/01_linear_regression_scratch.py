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
