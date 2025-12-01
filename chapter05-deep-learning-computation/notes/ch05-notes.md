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

### 初始化函数怎么施加

```python
def init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
```

`apply` 是递归的：先对 `net` 自己调 `init_normal`（它不是 `nn.Linear`，什么都不做），
再对每个子模块深度优先调用。所以 `init_fn` 里**必须**判断类型，
否则对 `ReLU` 调 `nn.init.normal_` 会报没有 `weight` 属性。

一个数字上的对照：`nn.Linear(4, 8)` 默认用
$U(-\sqrt{1/4}, \sqrt{1/4}) = U(-0.5, 0.5)$ 初始化，标准差是 $1/\sqrt{3}\times 0.5 \approx 0.289$；
用 `init_normal(std=0.01)` 之后标准差是 0.0099，小了 29 倍。
第 4 章说过初始化尺度直接决定梯度范数，所以这个差别不能忽略。

## 5.3 延后初始化

有一种尴尬情况：写网络的时候还不知道输入的维度。比如先搭好 20 层，
再看到数据是 784 维。`nn.LazyLinear(256)` 就是干这个的：创建时权重是
`UninitializedParameter`（形状未知），第一次 `forward` 时根据输入推断形状并调用
`reset_parameters()`。

```python
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.Linear(8, 2))
print(net[0].weight)        # <UninitializedParameter>
net(torch.rand(4, 4))       # 触发推断：输入 4 维 -> weight 形状 (8, 4)
print(net[0].weight.shape)  # torch.Size([8, 4])
```

**我的判断：这个概念重要，但 `LazyLinear` 我很少用。**
因为「不知道输入维度」通常说明我没有先把数据形状算清楚，
而形状算不清是后面所有 bug 的源头（第 6 章卷积尤其）。知道它在哪、
知道它为什么报 `has not been initialized` 就够了。

## 5.4 自定义层

### 无参层：CenteredLayer

```python
class CenteredLayer(nn.Module):
    def forward(self, X):
        return X - X.mean()
```

没有参数，但仍然要继承 `nn.Module`，因为要能塞进 `Sequential`、能被 `print` 出来、
能作为其他块的子模块。`parameters()` 返回空列表，这是合法的。

注意 `X.mean()` 是对**所有元素**求均值，不是按样本求均值。d2l 的例子就是全局均值，
用在「减去整体偏移」的场景。如果要按样本中心化，得写 `X.mean(dim=1, keepdim=True)`。
这两种我都在 `code/01` 里试了一遍，数值差别很明显。

### 带参层：MyLinear 与形状约定

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units) * 0.01)
        self.bias = nn.Parameter(torch.zeros(units))

    def forward(self, X):
        return X @ self.weight + self.bias
```

**形状约定是个真实的坑。** PyTorch 官方的 `nn.Linear(in, out)` 内部
`weight` 形状是 `(out, in)`，前向是 `X @ weight.T`；我习惯写 `(in, out)`，前向是 `X @ weight`。
两种都对，但一个文件里必须统一。第 3 章我混着用过一次，
结果 `state_dict` 加载时形状不匹配，报
`size mismatch for weight: copying a param with shape torch.Size([8, 4]) from checkpoint,
the shape in current model is torch.Size([4, 8])`。

### 初始化要自己写

`nn.Parameter(torch.randn(...))` 的默认尺度是标准正态，太大。
第 4 章测过，std=1.0 会让梯度范数到 1e3 量级。所以自定义层的 `__init__` 里
必须自己乘一个小的标准差，或者调用 `nn.init` 的 Xavier 版本。
我把这一步抽成了 `reset_parameters()`，和官方层的命名保持一致，方便 `apply` 复用。

## 5.5 读写文件

### state_dict 是什么

`net.state_dict()` 返回一个 `OrderedDict`，键是参数名（和 `named_parameters()` 一致），
值是张量（`detach()` 过的，不带计算图）。它**不含模型结构**，只有数值。

```python
print(net.state_dict().keys())
# odict_keys(['0.weight', '0.bias', '2.weight', '2.bias'])
```

### 为什么存字典而不是存整个对象

`torch.save(net, "model.pt")` 会把整个 Python 对象 pickle 下来，包括类的引用。
后果是：**改个类名、挪个文件位置，老存档就加载不了**，报
`Can't get attribute 'MyNet' on <module '__main__'>`。而且 pickle 反序列化本身有安全风险。

存 `state_dict` 就只要「结构 + 数值」两部分：结构由代码定义（可读、可 review），
数值由存档提供。这也是我现在所有 checkpoint 的做法：

```python
torch.save({"epoch": 12, "state_dict": net.state_dict(),
            "optimizer": opt.state_dict()}, "checkpoint/mlp.pt")
```

多存 `epoch` 和 `optimizer` 是为了断点续训——只存模型权重的话，
优化器的动量/二阶矩会丢，续训时前几步会抖一下（第 11 章讲 Adam 时会看到这个现象）。

### load_state_dict 的严格性

它默认 `strict=True`：键少了报 missing keys，键多了报 unexpected keys。
这个严格性是有意的，能挡住「结构改了但忘了改存档」的情况。
迁移学习时要故意忽略分类头，用 `strict=False` 并检查返回的
`missing_keys` / `unexpected_keys`，不要闭眼传 False。

### 设备问题

在有 GPU 的机器上存的存档，张量在 cuda 上；在没有 GPU 的机器上读会报
`Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`。
解决：`torch.load(path, map_location="cpu")`。这一条我写进 `code/03` 了，
虽然我现在只有 CPU，但存档迟早要在别的机器上读。
