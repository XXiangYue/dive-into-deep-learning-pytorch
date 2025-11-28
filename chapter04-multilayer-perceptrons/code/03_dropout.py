"""手写 dropout：自己实现置零逻辑，和一个不用 dropout 的版本对比。

网络结构 2 -> 64 -> 64 -> 2，比 code/01 更宽，这样在 200 个样本上更容易过拟合，
dropout 的效果才看得出来。

用法：python code/03_dropout.py

2025-11-28 修订：第一版训练时只置零、不做缩放，推理时原样输出，
两层叠加后激活被放大到 1/(1-p)^2 倍，验证损失反而比不用 dropout 更差（见 run-log）。
现在改成 inverted dropout：训练时除以 (1-p)，推理时原样，两边的期望一致。
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


def dropout_layer(X: torch.Tensor, dropout: float, is_training: bool) -> torch.Tensor:
    """inverted dropout。

    训练：按概率 dropout 置零，剩下的元素除以 (1-dropout)，
          这样输出的期望和推理时完全一致。
    推理：原样返回，不做任何缩放（inverted 的好处就是推理无额外操作）。

    注意 dropout=0 时必须短路：否则 mask 会在需要梯度的地方乘 0，浪费计算；
    dropout=1 时除以 0 会得到 inf，直接拒绝。
    """
    assert 0.0 <= dropout < 1.0, "dropout 概率必须在 [0, 1) 内"
    if not is_training or dropout == 0.0:
        return X
    # (rand > p) 得到 Bernoulli(1-p) 的 mask，均值是 1-p
    mask = (torch.rand(X.shape) > dropout).float()
    return X * mask / (1.0 - dropout)


def mlp(X: torch.Tensor, params: list, dropout: float = 0.0,
        is_training: bool = True) -> torch.Tensor:
    """前向：两层隐藏层，每层后接 ReLU 和 dropout。"""
    W1, b1, W2, b2, W3, b3 = params
    Z1 = X @ W1 + b1
    H1 = dropout_layer(torch.max(Z1, torch.zeros_like(Z1)), dropout, is_training)
    Z2 = H1 @ W2 + b2
    H2 = dropout_layer(torch.max(Z2, torch.zeros_like(Z2)), dropout, is_training)
    return H2 @ W3 + b3


def train(dropout: float, X: torch.Tensor, y: torch.Tensor,
          epochs: int = 60, lr: float = 0.1, batch_size: int = 20):
    """训练一个配置，返回 (训练损失, 训练准确率, 验证损失, 验证准确率)。

    这里用 8:2 切分做验证，验证集不参与梯度。
    """
    torch.manual_seed(42)
    params = init_mlp()
    n_train = int(len(X) * 0.8)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:], y[n_train:]

    for _ in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i: i + batch_size]
            loss = nn.functional.cross_entropy(
                mlp(X_tr[idx], params, dropout, is_training=True), y_tr[idx])
            loss.backward()
            with torch.no_grad():
                for p in params:
                    p -= lr * p.grad
                    p.grad.zero_()

    with torch.no_grad():
        # 验证时 is_training=False，不丢神经元也不缩放
        tr_logits = mlp(X_tr, params, dropout, is_training=False)
        va_logits = mlp(X_va, params, dropout, is_training=False)
        tr_loss = float(nn.functional.cross_entropy(tr_logits, y_tr))
        va_loss = float(nn.functional.cross_entropy(va_logits, y_va))
        tr_acc = float((tr_logits.argmax(1) == y_tr).float().mean())
        va_acc = float((va_logits.argmax(1) == y_va).float().mean())
    return tr_loss, tr_acc, va_loss, va_acc


if __name__ == "__main__":
    torch.manual_seed(42)
    X, y = make_data()
    print(f"{'dropout':>8}{'train_loss':>12}{'train_acc':>11}{'val_loss':>10}{'val_acc':>9}")
    for p in (0.0, 0.2, 0.5):
        tr_loss, tr_acc, va_loss, va_acc = train(p, X, y)
        print(f"{p:>8.1f}{tr_loss:>12.4f}{tr_acc:>11.3f}{va_loss:>10.4f}{va_acc:>9.3f}")

    # 验证「两种模式的输出期望一致」——这是本次修订的核心
    torch.manual_seed(42)
    params = init_mlp()
    Z1 = X[:5] @ params[0] + params[1]
    H1 = torch.max(Z1, torch.zeros_like(Z1))
    print()
    print("同一批隐藏层激活在两种模式下的均值（期望应接近）:")
    print("  训练模式:", round(float(dropout_layer(H1, 0.5, True).mean()), 4))
    print("  推理模式:", round(float(dropout_layer(H1, 0.5, False).mean()), 4))
