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


def build_net(num_inputs: int = 2) -> nn.Module:
    """单层线性回归：nn.Linear 内部就是 weight (1, d) 和 bias (1)。"""
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    # 手动初始化，和从零版保持一致（默认初始化是均匀分布，量级不同）
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0.0)
    return net


def train_concise(num_epochs: int, lr: float, batch_size: int,
                  features: torch.Tensor, labels: torch.Tensor):
    """简洁版训练循环：前向 → 算损失 → 清梯度 → 反传 → 更新。"""
    net = build_net(features.shape[1])
    loss = nn.MSELoss()        # 默认 reduction='mean'，已经除过 batch_size
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    loader = load_array(features, labels, batch_size)

    history = []
    for epoch in range(num_epochs):
        for X, y in loader:
            l = loss(net(X), y)     # 注意：这里不要再除 batch_size
            trainer.zero_grad()
            l.backward()
            trainer.step()
        with torch.no_grad():
            epoch_loss = float(loss(net(features), labels))
        history.append(epoch_loss)
        print(f"epoch {epoch + 1:>2}, loss {epoch_loss:.6f}")
    return net, history


if __name__ == "__main__":
    torch.manual_seed(42)
    features, labels = synthetic_data(TRUE_W, TRUE_B, 1000)

    net, history = train_concise(num_epochs=5, lr=0.03, batch_size=10,
                                 features=features, labels=labels)

    w_hat = net[0].weight.data.flatten().tolist()
    b_hat = net[0].bias.data.item()
    print()
    print("w 估计:", [round(v, 4) for v in w_hat], "真值:", TRUE_W.tolist())
    print("b 估计:", round(b_hat, 4), "真值:", TRUE_B)
    print("loss 首末:", round(history[0], 6), "->", round(history[-1], 6))
    print("参数量:", sum(p.numel() for p in net.parameters()),
          "（2 个权重 + 1 个偏置）")
