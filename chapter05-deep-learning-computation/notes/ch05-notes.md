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

## 5.1 层和块

### 一个块要干三件事

1. **存参数**：`nn.Parameter` 类型的属性会被登记到 `self._parameters`；
2. **算前向**：子类必须实现 `forward`，`__call__` 里会做一些 hook 相关的准备再调它；
3. **报结构**：`__repr__` 递归打印子模块树，`print(net)` 看到的就是这棵树。

第三点最容易被忽略。第 4 章我写 `MLPBlock` 时把两层压在一个 `forward` 里，
训练完全正常，但 `print(block)` 只有一行，`block.children()` 是空的，
既不能按层初始化，也不能按层查看梯度。**块的价值不在于封装，在于结构可见。**

### 自定义容器的关键

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)     # 关键在这一行

    def forward(self, X):
        for module in self._modules.values():
            X = module(X)
        return X
```

`add_module` 做了两件事：把模块写进 `self._modules`，并把它名字登记到 `_modules` 字典里。
如果写成 `self.layers = list(args)`，参数就完全不会被注册——
`parameters()` 返回空，`optim.SGD` 会报 `optimizer got an empty parameter list`。
这和第 5.2 节的坑是同一个机制。

### 嵌套

块里可以放块，所以 `net` 其实是一棵树。`parameters()` 是深度优先递归遍历整棵树，
`named_parameters()` 给每个参数一个带路径的名字，比如 `net.0.weight`。
**这个名字就是 `state_dict` 的键**，所以 5.5 节能直接对上——这是同一套机制的两面。

## 5.2 参数管理

### 三种写法的区别

```python
class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(3, 3))        # 注册 ✔
        self.b = [nn.Parameter(torch.randn(3, 3))]      # 不注册 ✘（list 里）
        self.c = nn.ParameterList([nn.Parameter(...)])  # 注册 ✔
        self.d = nn.ModuleList([nn.Linear(2, 2)])       # 注册 ✔
        self.e = nn.Linear(2, 2)                        # 注册 ✔
```

原因在 `nn.Module.__setattr__`：它只对几种类型做特殊处理——
`Parameter`、`Module`、`ParameterList` / `ModuleList` / `ModuleDict` 等容器。
赋值给一个普通 `list` 时，走的是 object 的默认 `__setattr__`，
`_parameters` 字典里什么都不会多。

**这个坑最阴的地方是它不报错。** `forward` 里照样能用 `self.b[0]`，
张量照样参与计算，梯度照样能算出来（`nn.Parameter` 本身就 `requires_grad=True`），
唯独 `optimizer` 拿不到它 → **这个参数永远不更新，而且没有任何警告。**
我在 `code/02` 里就是这么写的，跑完发现共享矩阵一个 epoch 没变过。

### 自查手段

```python
print(sum(p.numel() for p in net.parameters()))     # 参数量对不对
for name, p in net.named_parameters():              # 名字有没有缺
    print(name, tuple(p.shape), p.requires_grad)
```

我现在的习惯是：写完一个自定义块，先打印一遍 `named_parameters()`，
把参数量和手算的对一遍，再开始训练。这一步只花 10 秒，能省两小时。

### 参数共享

同一个 `nn.Module` 在 `Sequential` 里出现两次，参数只存一份：

```python
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 2))
```

`parameters()` 默认 `remove_duplicate=True`，所以 8→8 那层只算一次。
参数量：$4\times8+8 + 8\times8+8 + 8\times2+2 = 40 + 72 + 18 = 130$。
共享的好处是省参数、并且强制两组变换用同一个映射（类似权重绑定）。
