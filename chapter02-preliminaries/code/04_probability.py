"""概率部分的小实验：抽样、经验分布、大数定律。

三个函数分别回答：
1. 抽多少次之后，经验频率才接近真实概率？
2. 样本均值的偏差随样本量怎么衰减（应该是 O(1/sqrt(n))）？
3. 多项分布的经验分布与真实分布的 L1 距离是多少？
"""

import torch

# 固定种子，保证 run-log 里的数字可以复现
torch.manual_seed(42)


def sample_dice(n: int) -> torch.Tensor:
    """抽 n 次六面骰子，返回 0..5 的整数张量。"""
    return torch.randint(0, 6, (n,))


def empirical_freq(samples: torch.Tensor, k: int = 6) -> torch.Tensor:
    """经验频率：每个取值的出现次数 / 总数。用 bincount 一次算完，不要写 for。"""
    counts = torch.bincount(samples, minlength=k).float()
    return counts / samples.numel()


def law_of_large_numbers(sizes=(10, 100, 1000, 10000, 100000)) -> list:
    """对每个样本量，比较经验均值与理论均值 2.5（0..5 的均匀分布）。"""
    out = []
    for n in sizes:
        samples = sample_dice(n).float()
        mean = samples.mean().item()
        out.append((n, mean, abs(mean - 2.5)))
    return out


def multinomial_estimate(probs=(0.1, 0.2, 0.3, 0.4), n: int = 1000) -> dict:
    """按给定概率抽 n 次，返回经验分布与 L1 误差。"""
    p = torch.tensor(probs, dtype=torch.float32)
    samples = torch.multinomial(p, num_samples=n, replacement=True)
    freq = torch.bincount(samples, minlength=len(probs)).float() / n
    return {
        "true": [round(v, 4) for v in p.tolist()],
        "empirical": [round(v, 4) for v in freq.tolist()],
        "l1_error": round(float((freq - p).abs().sum()), 4),
    }
