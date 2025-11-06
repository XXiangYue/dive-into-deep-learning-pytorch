# 深度学习复现笔记（d2l + 西瓜书）

按两本教材的目录顺序，一章一章手写实现、跑实验、记笔记的仓库。
每条 commit 对应一次真实的学习动作：读完一节、写完一个函数、跑挂一次、修好一次。

## 仓库总览

| 路线 | 教材 | 章节 | 定位 |
|:---:|---|:---:|---|
| A | 《动手学深度学习》PyTorch 版（d2l） | 15 | 主线：从张量到 Transformer，代码为主 |
| B | 《机器学习》周志华（西瓜书） | 16 | 副线：公式推导与传统算法，补理论 |

两条线交错推进，不是读完一本再读另一本。理由：d2l 第 3 章的 softmax 回归
只给了交叉熵的梯度结论，西瓜书第 3 章从对数似然能把「为什么用交叉熵」推完整；
两边对上之后，公式才不是背下来的。

## 目录结构

```text
.
├── README.md                  # 本文件：总览、路线、环境
├── requirements.txt           # 依赖与版本约束
├── .gitignore                 # Python / Jupyter / checkpoint
├── chapter01-introduction/    # d2l 每章一个目录，名字照抄原书目录名
│   ├── README.md              # 本章知识点、目标、产出清单
│   ├── notes/                 # 读书笔记 + run-log（真实运行记录）
│   ├── code/                  # 可独立运行的脚本，01_ 02_ 表示阅读顺序
│   └── exercises/             # 课后习题与自己的解答
├── chapter02-preliminaries/
└── ...
```

每章内部固定四件套：`README.md`（要做什么）、`notes/`（读到了什么）、
`code/`（跑出来什么）、`exercises/`（还差什么）。

## 约定

- 提交信息前缀 `d2l-ch03:` / `ml-ch04:`，后面跟一句中文动宾短语。
- 一个脚本一个主题，不写大而全的 `main.py`。
- 需要数据就现场合成，仓库里不放数据集，也不联网下载。
- 随机种子统一 42，`notes/run-log.md` 记录当次的真实输出。
- 只用 CPU 也要能跑通；GPU 相关代码先判断 `torch.cuda.is_available()`。

## 环境

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
```

实际在用的机器：Windows 11 + Python 3.12 + torch 2.4 CPU 版，没有独显。
没有 GPU 也能走完 d2l 前十章，只是卷积那几章要调小 batch size、减少 epoch。
第 5 章会写设备迁移的代码，在没有 CUDA 的机器上它走 `cpu` 分支，不影响阅读。

## 进度

| 路线 | 章节 | 状态 |
|:---:|---|:---:|
| A | 1 引言 | 进行中 |
| A | 2~15 | 未开始 |
| B | 1~16 | 未开始 |
