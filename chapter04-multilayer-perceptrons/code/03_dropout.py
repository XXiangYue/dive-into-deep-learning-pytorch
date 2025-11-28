"""手写 dropout：自己实现置零逻辑，和一个不用 dropout 的版本对比。

网络结构 2 -> 64 -> 64 -> 2，比 code/01 更宽，这样在 200 个样本上更容易过拟合，
dropout 的效果才看得出来。

用法：python code/03_dropout.py
"""

import torch
from sklearn.datasets import make_moons
from torch import nn

NUM_HIDDENS = 64


def make_data(n_samples: int = 200, noise: float = 0.2, seed: int = 42):
    """和 code/01 完全一样的数据，方便对比。"""
    X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    X = (X - X.mean(dim=0, keepdim=True)) / X.std(dim=0, keepdim=True)
    return X, y


def init_mlp(num_inputs: int = 2, num_hiddens: int = NUM_HIDDENS, num_outputs: int = 2):
    """三个线性层 + 两个隐藏层，参数放进 list。"""
    def rand(shape, fan_in, fan_out):
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        return (torch.randn(shape) * std).requires_grad_(True)

    return [
        rand((num_inputs, num_hiddens), num_inputs, num_hiddens),
        torch.zeros(num_hiddens, requires_grad=True),
        rand((num_hiddens, num_hiddens), num_hiddens, num_hiddens),
        torch.zeros(num_hiddens, requires_grad=True),
        rand((num_hiddens, num_outputs), num_hiddens, num_outputs),
        torch.zeros(num_outputs, requires_grad=True),
    ]
