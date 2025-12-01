# 第 5 章 深度学习计算 · 读书笔记

## 这一章回答的问题

前四章我的代码是这样的：参数用 `list` 装、初始化手写、训练循环每章抄一遍、
模型存不下来。这一章把这些问题一次性解决。核心问题只有一个：
**框架怎么知道「这个模型的参数有哪些」？** 答案藏在 `nn.Module.__setattr__` 里。

## 框架树

```text
深度学习计算
├── 5.1 层和块
│   ├── 块的三个职责：存参数 / 算前向 / 报结构
│   ├── nn.Sequential 是最简单的容器
│   ├── 自定义容器：add_module + 遍历 _modules
│   └── 嵌套：块里可以有块（一棵树，不是列表）★
├── 5.2 参数管理
│   ├── parameters() / named_parameters()（默认去重）
│   ├── 只有「直接赋值」或 ParameterList/ModuleList 才会被注册 ★
│   ├── 嵌套块递归访问
│   └── 参数共享：同一个 Module 用两次
├── 5.3 延后初始化
│   ├── 不知道输入维度时怎么建层
│   ├── LazyLinear / UninitializedParameter
│   └── 第一次前向时推断形状（infer_parameters）
├── 5.4 自定义层
│   ├── 无参层：CenteredLayer
│   ├── 带参层：MyLinear（shape 是 (in, out) 还是 (out, in) 要统一）
│   └── apply(init_fn) 递归初始化
├── 5.5 读写文件
│   ├── state_dict：有序字典，「参数名 -> 张量」
│   ├── load_state_dict：按名字匹配，缺/多都会报错
│   └── 存对象 vs 存字典：对象依赖代码结构 ★
└── 5.6 / 5.7 GPU
    ├── torch.device、.to(device)、.cuda()
    ├── 模型和数据必须在同一个设备上
    └── torch.load(map_location=...) 解决「在有卡机器上存的、要在没卡机器上读」
```

标 ★ 的三处是这一章真正的知识点，其余都是查文档能解决的 API。

## 我的读法

5.1 和 5.2 是原理，必须读透（尤其是 `__setattr__` 那段）；
5.3 延后初始化知道有这回事就行，我很少用 `LazyLinear`；
5.4 ~ 5.6 全部动手写，因为 API 细节记不住，写一遍就有肌肉记忆了。
