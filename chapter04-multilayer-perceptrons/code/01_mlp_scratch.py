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
