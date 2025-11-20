"""线性回归的简洁实现：nn.Linear + DataLoader + MSELoss + optim.SGD。

和 01 用同一组合成数据与同样的超参，目的就是对照：
同样的训练效果下，简洁版少了哪些样板代码、又多了哪些隐含行为。

注意：为了脚本能独立运行，这里重复定义了一遍 synthetic_data 和真值。
文件名以数字开头没法 import，跨文件复用要改成包结构，第 3 章还不值得。
"""

import torch
from torch import nn
from torch.utils import data

TRUE_W = torch.tensor([2.0, -3.4])
TRUE_B = 4.2


def synthetic_data(w: torch.Tensor, b: float, num_examples: int, noise: float = 0.01):
    """和 01 完全相同的造数逻辑，保证两个脚本可比。"""
    X = torch.normal(0.0, 1.0, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0.0, noise, y.shape)
    return X, y.reshape((-1, 1))


def load_array(features: torch.Tensor, labels: torch.Tensor,
               batch_size: int, is_train: bool = True):
    """把张量包成 DataLoader。is_train 决定要不要每个 epoch 打乱。"""
    dataset = data.TensorDataset(features, labels)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
