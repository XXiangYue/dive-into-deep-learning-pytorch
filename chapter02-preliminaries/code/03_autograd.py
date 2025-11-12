"""自动微分验证：手算的偏导矩阵和 autograd 的结果是否一致。

核心想法：对向量函数 y = f(x)，雅可比矩阵的第 (i, j) 项是 ∂y_i/∂x_j。
朴素办法是对每个输出分量单独 backward 一次；autograd 的办法是传一个单位向量做
向量-雅可比积。两种都写一遍，方便对照。
"""

import torch


def f(x: torch.Tensor) -> torch.Tensor:
    """测试函数：y1 = x1^2 + 2*x2, y2 = x1 * x2, y3 = sin(x1)。

    手算雅可比（3x2）：
        [ 2*x1,   2    ]
        [  x2,   x1    ]
        [ cos(x1), 0   ]
    """
    return torch.stack([x[0] ** 2 + 2 * x[1], x[0] * x[1], torch.sin(x[0])])


def analytic_jacobian(x: torch.Tensor) -> torch.Tensor:
    """把上面的手算结果写成矩阵，用来和 autograd 比。"""
    return torch.tensor([
        [2 * x[0].item(), 2.0],
        [x[1].item(), x[0].item()],
        [torch.cos(x[0]).item(), 0.0],
    ])


def jacobian_by_backward(fn, x: torch.Tensor) -> torch.Tensor:
    """对每个输出分量单独 backward，取该分量的梯度作为雅可比的一行。"""
    rows = []
    for i in range(fn(x).numel()):
        xi = x.detach().clone().requires_grad_(True)
        y = fn(xi)
        # 只对第 i 个分量求导：把它当标量，其余分量乘 0
        y[i].backward()
        rows.append(xi.grad.detach().clone())
    return torch.stack(rows)


def jacobian_by_vjp(fn, x: torch.Tensor) -> torch.Tensor:
    """用向量-雅可比积：backward 传单位向量 e_i，一次拿到一行。"""
    rows = []
    m = fn(x).numel()
    for i in range(m):
        xi = x.detach().clone().requires_grad_(True)
        y = fn(xi)
        v = torch.zeros(m)
        v[i] = 1.0
        y.backward(v)          # 等价于 v^T J
        rows.append(xi.grad.detach().clone())
    return torch.stack(rows)
