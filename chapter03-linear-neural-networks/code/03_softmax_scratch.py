"""softmax 回归的从零实现：三类二维高斯团，不下载 Fashion-MNIST。

数据是三个中心、同一标准差的高斯团，线性可分但不是一维排开，
这样 softmax 的三个 logit 都有活干，能看出类别之间在互相竞争。

用法：python code/03_softmax_scratch.py
"""

import torch

NUM_CLASSES = 3
NUM_FEATURES = 2
# 三个类的中心，故意摆成三角形
CENTERS = torch.tensor([[2.0, 2.0], [-2.0, 1.5], [0.0, -2.5]])


def make_blobs(n_per_class: int = 300, std: float = 0.7, seed: int = 42):
    """生成三类二维点，返回 (X, y)：X (3n, 2) float32，y (3n,) int64。"""
    generator = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for k in range(NUM_CLASSES):
        # 注意 torch.normal 传张量做 mean 时不能再给 size，所以要 expand
        center = CENTERS[k].expand(n_per_class, NUM_FEATURES)
        xs.append(torch.normal(center, std, generator=generator))
        ys.append(torch.full((n_per_class,), k, dtype=torch.long))
    X = torch.cat(xs)
    y = torch.cat(ys)
    perm = torch.randperm(len(X), generator=generator)
    return X[perm], y[perm]


def standardize(X: torch.Tensor):
    """按特征标准化：softmax 的 logits 对特征尺度很敏感。"""
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    return (X - mean) / std


def init_params(num_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES):
    """权重用小方差正态、偏置置零，形状 (d, k)。"""
    W = torch.normal(0.0, 0.01, (num_features, num_classes), requires_grad=True)
    b = torch.zeros(num_classes, requires_grad=True)
    return W, b


def softmax(o: torch.Tensor) -> torch.Tensor:
    """按行归一化成概率：exp 之后除以行和。"""
    return torch.exp(o) / torch.exp(o).sum(axis=1, keepdim=True)


def net(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """线性前向，返回 logits，形状 (n, k)。"""
    return X @ W + b


def cross_entropy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """交叉熵：-log(正确类别的预测概率)，y_hat 是 softmax 之后的概率。"""
    rows = torch.arange(len(y_hat))
    return -torch.log(y_hat[rows, y])


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """准确率：预测类别（argmax）等于真实标签的比例。"""
    if y_hat.shape[1] > 1:
        pred = y_hat.argmax(axis=1)
    else:
        pred = y_hat
    return float((pred == y).float().mean())
