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


def fit(train, test, weight_decay: float = 0.0,
        epochs: int = 200, lr: float = 0.01):
    """训练一个线性层，返回 (net, 训练损失, 测试损失)。

    weight_decay 直接交给 optim.SGD：它会在梯度上自动加 lambda * w。
    注意这一版是对 net.parameters() 全体生效，包括 bias。
    """
    x_train, y_train = train
    x_test, y_test = test
    train_X, test_X = poly_features(x_train), poly_features(x_test)

    net = nn.Sequential(nn.Linear(DEGREE + 1, 1))
    net[0].weight.data.normal_(0.0, 0.01)
    net[0].bias.data.fill_(0.0)

    loss_fn = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        trainer.zero_grad()
        l = loss_fn(net(train_X), y_train)
        l.backward()
        trainer.step()

    with torch.no_grad():
        train_loss = float(loss_fn(net(train_X), y_train))
        test_loss = float(loss_fn(net(test_X), y_test))
    return net, train_loss, test_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    train_set, test_set = make_data()
    print(f"训练集 {NUM_TRAIN} 个点，测试集 {NUM_TEST} 个点，"
          f"特征维度 {DEGREE + 1}，参数量 {DEGREE + 2}")
    print()
    print(f"{'weight_decay':>14}{'train_loss':>12}{'test_loss':>12}{'gap':>10}")
    for wd in (0.0, 1e-4, 1e-2, 1e-1):
        net, tr, te = fit(train_set, test_set, weight_decay=wd)
        print(f"{wd:>14.0e}{tr:>12.4f}{te:>12.4f}{te - tr:>10.4f}")

    # 权重范数：不衰减 vs 1e-2，看正则把权重压小了多少
    print()
    for wd in (0.0, 1e-2):
        net_wd, _, _ = fit(train_set, test_set, weight_decay=wd)
        norm = float(net_wd[0].weight.data.norm())
        print(f"wd={wd:.0e} 权重 L2 范数: {norm:.2f}")
    net, _, _ = fit(train_set, test_set, weight_decay=0.0)
    print("不衰减时前 6 个权重:",
          [round(v, 2) for v in net[0].weight.data.flatten()[:6].tolist()])
