"""自定义层：先写两个最小的层，再写容器和整块网络。

CenteredLayer 演示「无参数但有前向」的层；
MyLinear 演示「参数必须用 nn.Parameter 包装」；
MySequential 演示容器怎么写（关键在 add_module）；
MLPBlock 演示把一个网络写成一个块。

用法：python code/01_custom_layers.py
"""

import torch
from torch import nn
from torch.nn import functional as F


class CenteredLayer(nn.Module):
    """减去全局均值的层：没有参数，但仍然是一个块。"""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X - X.mean()


class MyLinear(nn.Module):
    """手写全连接层。权重形状用 (in_units, units)，前向就是 X @ W。

    注意官方 nn.Linear 的 weight 是 (out, in)，前向是 X @ W.T。
    两种约定都对，但一个文件里必须统一，否则 state_dict 会形状不匹配。
    """

    def __init__(self, in_units: int, units: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units) * 0.01)
        self.bias = nn.Parameter(torch.zeros(units))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.weight + self.bias
