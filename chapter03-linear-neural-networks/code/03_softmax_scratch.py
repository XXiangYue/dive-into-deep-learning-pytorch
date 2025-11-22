"""softmax 回归的从零实现：三类二维高斯团，不下载 Fashion-MNIST。

数据是三个中心、同一标准差的高斯团，线性可分但不是一维排开，
这样 softmax 的三个 logit 都有活干，能看出类别之间在互相竞争。

用法：python code/03_softmax_scratch.py

2025-11-22 修订：原版 softmax 直接算 exp(o)，lr 稍大 logit 到 120 就 exp 溢出成 inf，
loss 变 nan（见 run-log）。现在改成先减每行最大值，并且交叉熵直接吃 logits、
用 log-sum-exp 的形式算，全程不出现「先 softmax 再 log」的中间量。
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


def net(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """线性前向，返回 logits，形状 (n, k)。"""
    return X @ W + b


def softmax(o: torch.Tensor) -> torch.Tensor:
    """按行归一化成概率。

    先减去每行最大值：数学上完全等价（分子分母同乘 e^{-max}），
    但指数最大只有 exp(0)=1，不会溢出成 inf。
    """
    o = o - o.max(dim=1, keepdim=True).values
    exp = torch.exp(o)
    return exp / exp.sum(dim=1, keepdim=True)


def log_softmax(o: torch.Tensor) -> torch.Tensor:
    """log(softmax(o)) 的稳定写法，交叉熵直接用这个。"""
    shifted = o - o.max(dim=1, keepdim=True).values
    # log-sum-exp 的位移版本：ln sum exp(shifted) 一定 <= ln k，不会炸
    return shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))


def cross_entropy(o: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """交叉熵：直接吃 logits，内部用 log_softmax，避免先 softmax 再 log 的精度损失。"""
    rows = torch.arange(len(o))
    return -log_softmax(o)[rows, y]


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """准确率：预测类别（argmax）等于真实标签的比例。"""
    if y_hat.shape[1] > 1:
        pred = y_hat.argmax(axis=1)
    else:
        pred = y_hat
    return float((pred == y).float().mean())


def train_softmax(num_epochs: int, lr: float, batch_size: int,
                  params: list, X_train: torch.Tensor, y_train: torch.Tensor,
                  X_val: torch.Tensor, y_val: torch.Tensor) -> list:
    """训练 softmax 回归，每个 epoch 打印训练损失与两边准确率。"""
    W, b = params
    history = []
    n = len(X_train)
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        X_shuf, y_shuf = X_train[perm], y_train[perm]
        for i in range(0, n, batch_size):
            Xb = X_shuf[i: i + batch_size]
            yb = y_shuf[i: i + batch_size]
            l = cross_entropy(net(Xb, W, b), yb)
            l.sum().backward()
            with torch.no_grad():
                W -= lr * W.grad / len(Xb)
                b -= lr * b.grad / len(Xb)
                W.grad.zero_()
                b.grad.zero_()
        with torch.no_grad():
            train_loss = float(cross_entropy(net(X_train, W, b), y_train).mean())
            train_acc = accuracy(net(X_train, W, b), y_train)
            val_acc = accuracy(net(X_val, W, b), y_val)
        history.append((epoch + 1, train_loss, train_acc, val_acc))
        print(f"epoch {epoch + 1:>2}  loss {train_loss:.4f}  "
              f"train acc {train_acc:.3f}  val acc {val_acc:.3f}")
    return history


if __name__ == "__main__":
    torch.manual_seed(42)
    X, y = make_blobs(n_per_class=300, std=0.7)
    X = standardize(X)
    # 800 / 100 切分，验证集不参与训练
    X_train, y_train = X[:800], y[:800]
    X_val, y_val = X[800:], y[800:]

    W, b = init_params()
    history = train_softmax(num_epochs=10, lr=0.1, batch_size=32,
                            params=[W, b], X_train=X_train, y_train=y_train,
                            X_val=X_val, y_val=y_val)
    print()
    print("最终 W:\n", W.detach())
    print("最终 b:", b.detach().tolist())
    print("验证集准确率:", round(history[-1][3], 4))

    # 压力测试：减最大值之后，logit 再大也不会溢出
    big = torch.tensor([[1000.0, 0.0, 0.0]])
    print("大 logits 的 softmax:", softmax(big))
    print("大 logits 的交叉熵（y=0）:", round(float(cross_entropy(big, torch.tensor([0]))), 6))
    print("大 logits 的交叉熵（y=1）:", round(float(cross_entropy(big, torch.tensor([1]))), 4))
