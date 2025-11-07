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


def build_plan(hours_per_week: float = 10.0) -> list:
    """按累计工时推算每章的起始周。

    不能「一章塞一周」：第 7 章预估 22 小时，一周只有 10 小时，
    硬塞的结果是每周都超支、后面的周次全是假的。
    所以用「已花工时 // 每周预算 + 1」算起始周，一章自然就跨了好几周。
    """
    plan = []
    spent = 0.0
    for chapter, directory, hours in CHAPTERS:
        start_week = int(spent // hours_per_week) + 1
        plan.append(PlanItem(chapter=chapter, directory=directory,
                             hours=float(hours), week=start_week))
        spent += hours
    return plan


def summarize(plan: list, hours_per_week: float = 10.0) -> dict:
    """汇总总工时、读完需要的周数、最重的三章。"""
    total = sum(item.hours for item in plan)
    heaviest = sorted(plan, key=lambda it: it.hours, reverse=True)[:3]
    longest = max(plan, key=lambda it: it.hours)
    return {
        "chapters": len(plan),
        "total_hours": total,
        "weeks_to_finish": math.ceil(total / hours_per_week),
        "avg_hours_per_chapter": round(total / len(plan), 2),
        "longest_chapter": (longest.chapter, longest.hours),
        "heaviest": [(it.chapter, it.hours) for it in heaviest],
    }


def render_markdown(plan: list, hours_per_week: float = 10.0) -> str:
    """渲染成 Markdown 表格：起始周 + 预计占用周数。

    不用 DataFrame.to_markdown，那个要装 tabulate，多一个依赖不值。
    """
    lines = ["| 章 | 目录 | 工时 | 起始周 | 占用周数 |",
             "|:---:|---|---:|---:|---:|"]
    for item in plan:
        span = math.ceil(item.hours / hours_per_week)
        lines.append(
            f"| {item.chapter} | {item.directory} | {item.hours:.0f} "
            f"| {item.week} | {span} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    plan = build_plan(hours_per_week=10.0)
    print(render_markdown(plan, hours_per_week=10.0))
    print()
    stats = summarize(plan, hours_per_week=10.0)
    for key, value in stats.items():
        print(f"{key:>24}: {value}")
