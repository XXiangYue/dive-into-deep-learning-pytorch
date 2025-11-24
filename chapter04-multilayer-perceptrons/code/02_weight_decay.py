"""权重衰减：先用高阶多项式特征把过拟合造出来，再把它压回去。

设定：
- 真实函数是三次多项式 y = 5 + 1.2x - 3.4x^2 + 5.6x^3，噪声 0.05；
- 训练集只有 20 个点，测试集 100 个点；
- 模型把输入摊成 21 维特征 [1, x, x^2, ..., x^20]，再接一个线性层。
  21 个参数 vs 20 个样本，几乎可以插值，过拟合是必然的。

用法：python code/02_weight_decay.py
"""

import torch
from torch import nn

NUM_TRAIN, NUM_TEST, DEGREE = 20, 100, 20


def true_fn(x: torch.Tensor) -> torch.Tensor:
    """真实的三次多项式。系数故意有大有小，这样高阶项更容易作妖。"""
    return 5.0 + 1.2 * x - 3.4 * x ** 2 + 5.6 * x ** 3


def make_data(n_train: int = NUM_TRAIN, n_test: int = NUM_TEST,
              noise: float = 0.05, seed: int = 42):
    """造训练集和测试集，返回 ((x_train, y_train), (x_test, y_test))。"""
    torch.manual_seed(seed)

    def one(n):
        x = torch.normal(0.0, 1.0, (n, 1))
        y = true_fn(x) + torch.normal(0.0, noise, (n, 1))
        return x, y

    return one(n_train), one(n_test)


def poly_features(x: torch.Tensor, degree: int = DEGREE) -> torch.Tensor:
    """把 (n, 1) 摊成 (n, degree+1)：[1, x, x^2, ...]。"""
    return torch.cat([x ** i for i in range(degree + 1)], dim=1)
