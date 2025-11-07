"""把 d2l 的 15 章排成一张可执行的学习计划表。

为什么第 1 章就写这个：这一章没有模型可训，最现实的问题是「时间够不够」。
计划表按每周可用工时把 15 章摊到日历上，输出 Markdown，直接贴进 run-log。
所有数据写死在 CHAPTERS 里，不读外部文件。
"""

import math
from dataclasses import dataclass

# (章号, 目录名, 预估工时)。工时是我通读一遍后的体感估计：
# 前 5 章偏基础但要从零实现，第 6~10 章每个模型都要画形状，给得更宽。
CHAPTERS = [
    (1, "chapter01-introduction", 4),
    (2, "chapter02-preliminaries", 12),
    (3, "chapter03-linear-neural-networks", 14),
    (4, "chapter04-multilayer-perceptrons", 16),
    (5, "chapter05-deep-learning-computation", 10),
    (6, "chapter06-convolutional-neural-networks", 18),
    (7, "chapter07-modern-convolutional-neural-networks", 22),
    (8, "chapter08-recurrent-neural-networks", 16),
    (9, "chapter09-modern-recurrent-neural-networks", 18),
    (10, "chapter10-attention-mechanisms", 20),
    (11, "chapter11-optimization-algorithms", 12),
    (12, "chapter12-computational-performance", 8),
    (13, "chapter13-computer-vision", 16),
    (14, "chapter14-natural-language-processing-pretraining", 18),
    (15, "chapter15-natural-language-processing-applications", 14),
]


@dataclass
class PlanItem:
    """一章的计划条目。week 由 build_plan 填。"""

    chapter: int
    directory: str
    hours: float
    week: int = 0
