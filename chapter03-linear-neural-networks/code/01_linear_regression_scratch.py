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


def sgd(params: list, lr: float, batch_size: int) -> None:
    """小批量随机梯度下降：用当前批量的梯度原地更新参数。

    梯度已经在 backward 时按批量求和过，所以要除以 batch_size
    才能让更新幅度和批量大小无关。
    """
    for param in params:
        param -= lr * param.grad / batch_size
        param.grad.zero_()


def train(num_epochs: int, lr: float, batch_size: int, params: list,
          features: torch.Tensor, labels: torch.Tensor) -> list:
    """训练若干 epoch，返回每个 epoch 结束时的全量平均损失。"""
    history = []
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            loss = squared_loss(linreg(X, params[0], params[1]), y)
            loss.sum().backward()      # 对批量求和再反传，梯度也是按批求和
            sgd(params, lr, batch_size)
        with torch.no_grad():
            epoch_loss = float(squared_loss(linreg(features, params[0], params[1]), labels).mean())
        history.append(epoch_loss)
        print(f"epoch {epoch + 1:>2}, loss {epoch_loss:.6f}")
    return history


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    features, labels = synthetic_data(TRUE_W, TRUE_B, 1000)
    # 参数用很小的随机数初始化，b 用 0。初始值太大的话前几个 batch 的损失会爆
    w = torch.normal(0.0, 0.01, (2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    history = train(num_epochs=5, lr=0.03, batch_size=10,
                    params=[w, b], features=features, labels=labels)

    print()
    print("w 估计:", [round(v, 4) for v in w.detach().flatten().tolist()],
          "真值:", TRUE_W.tolist())
    print("b 估计:", round(b.detach().item(), 4), "真值:", TRUE_B)
    print("loss 首末:", round(history[0], 6), "->", round(history[-1], 6))

    # 用正规方程做基准：如果 SGD 写对了，两者的差应该在 1e-3 以内
    exact = analytic_solution(features, labels)
    print("解析解 [w1, w2, b]:", [round(v, 4) for v in exact.tolist()])
