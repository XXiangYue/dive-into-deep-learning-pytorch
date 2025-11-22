# 第 3 章 运行记录

环境同前：Windows 11 / Python 3.12.10 / torch 2.4.1+cpu。

## 2025-11-22 14:20 线性回归从零版：第一次直接报错

```bash
python code/01_linear_regression_scratch.py
```

```text
Traceback (most recent call last):
  File "code/01_linear_regression_scratch.py", line 71, in <module>
    history = train(num_epochs=5, lr=0.03, batch_size=10, ...)
  ...
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

`param -= lr * param.grad / batch_size` 这一行触发的，见笔记 3.2 的坑二。
当时先试了 `param.data -= ...`，能跑，但我不确定这是不是正确做法，
查了文档才知道推荐写法是包 `torch.no_grad()`。这一版先这么跑：

```text
epoch  1, loss 0.043081
epoch  2, loss 0.001756
epoch  3, loss 0.000312
epoch  4, loss 0.000117
epoch  5, loss 0.000098
w 估计: [2.0003, -3.4002] 真值: [2.0, -3.4]
b 估计: 4.1998 真值: 4.2
loss 首末: 0.043081 -> 9.8e-05
解析解 [w1, w2, b]: [2.0002, -3.4001, 4.1998]
```

**数字怎么读：**

- 噪声标准差 0.01，理论上损失下界是 $0.5\sigma^2 = 5\times10^{-5}$，
  跑出 9.8e-5，同量级，说明已经到极限了，不是没收敛；
- SGD 结果和正规方程给的点估计在小数点后 3 位一致，说明手写的前向和梯度都对；
- loss 从 0.043 掉到 1e-4 跨了三个数量级，前两个 epoch 就完成了 99% 的下降，
  这是线性回归（凸、条件数好）的典型特征，别指望神经网络也这样。
