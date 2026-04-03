# PyTorch Cheat Sheet (CS5242 Labs)

---

## 1. Tensor 创建

### `torch.Tensor(data)` — 从 Python 列表创建（默认 FloatTensor）

```python
import torch

a = torch.Tensor([5.3, 2.1, -3.1])        # 1D 向量
A = torch.Tensor([[1, 2], [3, 4]])         # 2×2 矩阵
# a.type() → 'torch.FloatTensor'（无论输入是否为整数）
```

### `torch.tensor(data)` — 自动推断类型

```python
x = torch.tensor([[1, 2], [3, 4]])         # 整数输入 → LongTensor
y = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # 浮点输入 → FloatTensor
```

### `torch.rand(*sizes)` — 均匀分布 [0, 1)

```python
x = torch.rand(10, 2)      # 10×2 矩阵，元素在 [0,1) 之间
x = torch.rand(5, 2, 3)    # 5×2×3 三维张量
```

### `torch.zeros(*sizes)` — 全零张量

```python
x = torch.zeros(10, 2)       # 10×2 全零矩阵
x = torch.zeros(2, 3, 4, 5)  # 4维全零张量
```

### `torch.arange(n)` — 等差序列 0 到 n-1

```python
x = torch.arange(10)  # tensor([0, 1, 2, ..., 9])，类型为 LongTensor
```

### `torch.randperm(n)` — 0 到 n-1 的随机排列（用于 epoch 训练中打乱数据）

```python
shuffled_indices = torch.randperm(60000)  # 打乱 60000 个样本的索引
# 例: tensor([48293, 12, 59001, ...])
```

### `torch.LongTensor(data)` — 整数张量（用于标签和索引）

```python
labels = torch.LongTensor([2, 3])  # 类别标签

# 生成随机索引用于 minibatch 采样
bs = 64
indices = torch.LongTensor(bs).random_(0, 60000)  # 64 个随机索引，范围 [0, 60000)
```

---

## 2. Tensor 属性查看

### `.size()` / `.size(dim)` — 查看形状

```python
x = torch.rand(5, 2, 3)
x.size()    # torch.Size([5, 2, 3])  — 完整形状
x.size(0)   # 5  — 第 0 维大小
x.size(1)   # 2  — 第 1 维大小
```

### `.dim()` — 维度数量

```python
A = torch.rand(3, 4)
A.dim()   # 2  — 矩阵是 2 维
v = torch.rand(5)
v.dim()   # 1  — 向量是 1 维
s = torch.tensor(3.14)
s.dim()   # 0  — 标量是 0 维
```

### `.type()` — 数据类型

```python
x = torch.Tensor([1, 2, 3])
x.type()  # 'torch.FloatTensor'

y = torch.LongTensor([1, 2])
y.type()  # 'torch.LongTensor'
```

---

## 3. 类型转换

### `.float()` / `.long()` — 类型转换

```python
x = torch.LongTensor([1, 2, 3])
x_float = x.float()   # LongTensor → FloatTensor（用于送入网络计算）

y = torch.Tensor([1.5, 2.7])
y_long = y.long()     # FloatTensor → LongTensor（用于作为索引或标签）
```

### `.item()` — 将 0 维 Tensor 转为 Python 数值

```python
a = torch.tensor(8)
b = a.item()    # b = 8，类型为 int

loss_val = loss.detach().item()  # 训练中提取 loss 的标量值用于打印
```

---

## 4. 形状变换

### `.view(*sizes)` — 重塑张量（不改变数据，不复制）

```python
x = torch.arange(10)      # tensor([0, 1, ..., 9])，shape: [10]
y = x.view(2, 5)          # shape: [2, 5] — 重塑为 2 行 5 列
z = x.view(5, 2)          # shape: [5, 2] — 重塑为 5 行 2 列

# MNIST: 28×28 图片展平送入全连接层
inputs = minibatch_data.view(bs, 784)       # [bs, 28, 28] → [bs, 784]

# CIFAR: 3×32×32 图片展平
inputs = minibatch_data.view(bs, 3072)      # [bs, 3, 32, 32] → [bs, 3072]

# 单张图片预测
scores = net(image.view(1, 784))            # [28, 28] → [1, 784]
```

---

## 5. 索引与切片

### 基本索引

```python
A = torch.tensor([[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]])

A[0]        # tensor([10, 20, 30])  — 第 0 行
A[2]        # tensor([70, 80, 90])  — 第 2 行
A[0, 1]     # tensor(20)           — 第 0 行第 1 列元素
```

### 切片

```python
A[0:2]      # 前 2 行: tensor([[10,20,30], [40,50,60]])
A[1:3]      # 第 1 到第 2 行

# 取 minibatch 数据
minibatch_data  = train_data[idx:idx+bs]    # 从 idx 开始取 bs 个样本
minibatch_label = train_label[idx:idx+bs]
```

### 高级索引（用索引张量取样本）

```python
# 随机采样方式构建 minibatch
indices = torch.LongTensor(bs).random_(0, 60000)
minibatch_data  = train_data[indices]     # 用索引张量取出对应样本
minibatch_label = train_label[indices]

# epoch 训练中按打乱顺序取
shuffled_indices = torch.randperm(60000)
for count in range(0, 60000, bs):
    indices = shuffled_indices[count:count+bs]
    minibatch_data  = train_data[indices]
    minibatch_label = train_label[indices]
```

---

## 6. 数学运算

### 逐元素运算

```python
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

C = A + B       # 逐元素加法
D = A * B       # 逐元素乘法（Hadamard 积）
E = 2 * A       # 标量乘法
```

### 矩阵乘法

```python
F = A @ B           # 矩阵乘法（推荐写法）
G = torch.mm(A, B)  # 等价的矩阵乘法
```

### 归约运算

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
x.sum()     # tensor(10.)  — 求和
x.min()     # tensor(1.)   — 最小值
x.max()     # tensor(4.)   — 最大值
```

### 常用数学函数

```python
prob = torch.tensor([0.2, 0.5, 0.3])
log_prob = torch.log(prob)     # 取对数（用于 NLLLoss）

scores = torch.tensor([[2.0, 1.0, 0.1]])
pred = scores.argmax(dim=1)   # tensor([0])  — 返回最大值的索引（预测类别）
```

---

## 7. 神经网络模块 (`nn.Module`)

### 自定义网络的基本结构

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNet, self).__init__()
        # 定义网络层
        self.layer1 = nn.Linear(input_size, hidden_size, bias=True)
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, x):
        # 定义前向传播
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

net = MyNet(784, 50, 10)
```

### `nn.Linear(in_features, out_features, bias=True)` — 全连接层

```python
# 参数说明:
#   in_features  — 输入特征数
#   out_features — 输出特征数
#   bias         — 是否加偏置（默认 True）

fc = nn.Linear(784, 10, bias=False)    # 784 → 10，无偏置
fc = nn.Linear(784, 10)               # 784 → 10，有偏置

# 查看权重和偏置
fc.weight       # shape: [10, 784]
fc.bias         # shape: [10] 或 None（bias=False 时）

# 手动修改权重（需在 no_grad 上下文中）
with torch.no_grad():
    fc.weight[0, 0] = 0
    fc.weight[0, 1] = 1
```

### `net.parameters()` — 获取所有可学习参数

```python
net = MyNet(784, 50, 10)

# 传给优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# 统计参数数量
total = sum(p.numel() for p in net.parameters())
print(f'Total parameters: {total}')
```

---

## 8. 激活函数

### `torch.relu(x)` — ReLU: max(0, x)

```python
x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
y = torch.relu(x)  # tensor([0., 0., 1., 2.])

# 在网络 forward 中使用
def forward(self, x):
    x = self.layer1(x)
    x = torch.relu(x)     # 每个隐藏层后加 ReLU
    x = self.layer2(x)
    x = torch.relu(x)
    x = self.layer3(x)    # 输出层不加 ReLU
    return x
```

### `torch.softmax(x, dim)` — Softmax（转为概率分布）

```python
# 参数说明:
#   x   — 输入张量（logits / scores）
#   dim — 在哪个维度做 softmax（概率在该维度上和为 1）

x = torch.tensor([2.0, 1.0, 0.1])
prob = torch.softmax(x, dim=0)  # tensor([0.6590, 0.2424, 0.0986])，sum=1

# 对 batch 的 scores 做 softmax
scores = torch.rand(64, 10)             # batch=64, 10 个类
prob = torch.softmax(scores, dim=1)     # 每个样本的 10 个类概率和为 1
```

---

## 9. 损失函数

### `nn.NLLLoss()` — 负对数似然损失（需要 log-probabilities 作为输入）

```python
criterion = nn.NLLLoss()

# 使用流程: scores → softmax → log → NLLLoss
scores = net(inputs)                    # 网络输出 raw scores
prob = torch.softmax(scores, dim=1)     # 转为概率
log_prob = torch.log(prob)              # 取对数
loss = criterion(log_prob, labels)      # labels: LongTensor，类别索引
```

### `nn.CrossEntropyLoss()` — 交叉熵损失（直接接受 raw logits，内部自动做 softmax）

```python
criterion = nn.CrossEntropyLoss()

# 使用流程: scores → CrossEntropyLoss（更简洁，推荐使用）
scores = net(inputs)                    # 网络输出 raw scores
loss = criterion(scores, labels)        # labels: LongTensor，类别索引
# 注意: 不需要手动做 softmax！
```

---

## 10. 优化器

### `torch.optim.SGD` — 随机梯度下降

```python
# 参数说明:
#   params — 模型参数（来自 net.parameters()）
#   lr     — 学习率

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# 训练中使用的三步曲:
optimizer.zero_grad()   # 1. 清零梯度（每个 batch 必须做）
loss.backward()         # 2. 反向传播计算梯度
optimizer.step()        # 3. 更新参数: W = W - lr * dL/dW
```

### 学习率衰减（手动调度）

```python
my_lr = 0.05
for epoch in range(200):
    # 每 10 个 epoch 衰减一次
    if epoch % 10 == 0 and epoch > 10:
        my_lr = my_lr / 1.5
    optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)
    # ... 训练循环 ...
```

---

## 11. 梯度控制

### `requires_grad_()` — 开启梯度追踪

```python
inputs = minibatch_data.view(bs, 784)
inputs.requires_grad_()   # 告诉 PyTorch 追踪对 inputs 的所有操作
```

### `torch.no_grad()` — 禁用梯度计算（用于推理和手动修改权重）

```python
# 推理时不需要计算梯度，节省内存
with torch.no_grad():
    scores = net(test_inputs)

# 手动修改权重时需要
with torch.no_grad():
    net.layer1.weight[0, 0] = 0
```

### `.detach()` — 从计算图分离

```python
# 提取 loss 用于记录（不保留计算图）
running_loss += loss.detach().item()

# 计算准确率时分离 scores
error = get_error(scores.detach(), labels)
```

### `loss.backward()` — 反向传播

```python
loss = criterion(scores, labels)
loss.backward()   # 计算所有参数的梯度 dL/dW
```

---

## 12. 数据 I/O

### `torch.load` / `torch.save` — 加载与保存张量

```python
# 加载预处理好的数据
train_data  = torch.load('data/mnist/train_data.pt')    # shape: [60000, 28, 28]
train_label = torch.load('data/mnist/train_label.pt')   # shape: [60000]
test_data   = torch.load('data/mnist/test_data.pt')     # shape: [10000, 28, 28]
test_label  = torch.load('data/mnist/test_label.pt')    # shape: [10000]

# 保存张量
torch.save(train_data, 'data/mnist/train_data.pt')
```

### `.numpy()` — 转为 NumPy 数组（用于绘图）

```python
import numpy as np

arr = x.numpy()                             # Tensor → ndarray
img = np.transpose(x.numpy(), (1, 2, 0))   # [C, H, W] → [H, W, C]（用于 matplotlib）
```

---

## 13. 评估指标

### 计算分类错误率

```python
def get_error(scores, labels):
    """计算 batch 的错误率"""
    bs = scores.size(0)                        # batch size
    predicted_labels = scores.argmax(dim=1)    # 取每行最大值的索引作为预测类别
    indicator = (predicted_labels == labels)    # 逐元素比较
    num_matches = indicator.sum()              # 正确预测数
    return 1 - num_matches.float() / bs        # 错误率 = 1 - 准确率
```

---

## 14. 完整训练流程

### 方式一：随机采样 Minibatch（简单版）

```python
net = MyNet(784, 50, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
bs = 64

for iteration in range(5000):
    # 随机取 minibatch
    indices = torch.LongTensor(bs).random_(0, 60000)
    minibatch_data  = train_data[indices]
    minibatch_label = train_label[indices]
    inputs = minibatch_data.view(bs, 784)

    # 前向传播
    inputs.requires_grad_()
    scores = net(inputs)
    loss = criterion(scores, minibatch_label)

    # 反向传播 + 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 方式二：Epoch 训练 + 监控 Loss/Error（推荐）

```python
net = MyNet(784, 50, 10)
criterion = nn.CrossEntropyLoss()
bs = 64
num_epochs = 30

my_lr = 0.05
for epoch in range(num_epochs):
    # 学习率衰减
    if epoch % 10 == 0 and epoch > 10:
        my_lr = my_lr / 1.5
    optimizer = torch.optim.SGD(net.parameters(), lr=my_lr)

    # 每个 epoch 打乱数据
    shuffled_indices = torch.randperm(60000)
    
    running_loss = 0.0
    running_error = 0.0
    num_batches = 0

    for count in range(0, 60000, bs):
        # 取当前 minibatch
        indices = shuffled_indices[count:count+bs]
        minibatch_data  = train_data[indices]
        minibatch_label = train_label[indices]
        inputs = minibatch_data.view(bs, 784)

        # 前向
        inputs.requires_grad_()
        scores = net(inputs)
        loss = criterion(scores, minibatch_label)

        # 反向 + 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 监控
        running_loss += loss.detach().item()
        running_error += get_error(scores.detach(), minibatch_label)
        num_batches += 1

    # 打印每个 epoch 的平均 loss 和 error
    avg_loss  = running_loss / num_batches
    avg_error = running_error / num_batches
    print(f'epoch={epoch}  loss={avg_loss:.4f}  error={avg_error*100:.2f}%')
```

### 测试集评估

```python
def eval_on_test_set(net, test_data, test_label, bs=64):
    running_error = 0.0
    num_batches = 0

    for i in range(0, 10000, bs):
        minibatch_data  = test_data[i:i+bs]
        minibatch_label = test_label[i:i+bs]
        inputs = minibatch_data.view(bs, 784)

        scores = net(inputs)
        error = get_error(scores.detach(), minibatch_label)
        running_error += error.item()
        num_batches += 1

    total_error = running_error / num_batches
    print(f'Test set error: {total_error*100:.2f}%')
```

---

## 15. 易混淆 & 易错点

### 15.1 `torch.Tensor()` vs `torch.tensor()` — 类型推断不同

```python
# torch.Tensor() — 始终创建 FloatTensor，即使传入整数
a = torch.Tensor([1, 2, 3])
a.type()    # 'torch.FloatTensor' ← 整数被强制转为浮点

# torch.tensor() — 根据输入数据自动推断类型
b = torch.tensor([1, 2, 3])
b.type()    # 'torch.LongTensor' ← 整数输入 → LongTensor

c = torch.tensor([1.0, 2.0, 3.0])
c.type()    # 'torch.FloatTensor' ← 浮点输入 → FloatTensor

# ⚠ 常见错误: 用 torch.Tensor 创建标签，导致类型不对
labels = torch.Tensor([0, 1, 2])        # ✗ FloatTensor，不能用于 CrossEntropyLoss
labels = torch.tensor([0, 1, 2])        # ✓ LongTensor，可以直接用
labels = torch.LongTensor([0, 1, 2])    # ✓ 显式指定 LongTensor
```

### 15.2 `random_()` vs `randperm()` — 有放回 vs 无放回

```python
# random_(low, high) — 有放回采样，范围 [low, high)，high 不包含
idx = torch.LongTensor(5).random_(0, 100)
# 可能出现重复: tensor([42, 7, 42, 88, 3])
# 范围: 0, 1, 2, ..., 99（不含 100）

# randperm(n) — 无放回排列，范围 [0, n)，n 不包含
idx = torch.randperm(100)
# 0~99 的随机排列，不会重复: tensor([47, 3, 91, ...])
# 取前 5 个等价于从 100 个中无放回抽 5 个: idx[:5]

# ⚠ 易错: random_ 的上界不包含，randperm 的参数 n 也不包含
torch.LongTensor(3).random_(0, 10)   # 生成 0~9，不会出现 10
torch.randperm(10)                    # 生成 0~9 的排列，不会出现 10
```

### 15.3 各种"范围"的边界总结 — 左闭右开是主流

```python
# torch.arange(start, end) — [start, end)，不包含 end
torch.arange(5)         # tensor([0, 1, 2, 3, 4])，不含 5
torch.arange(2, 7)      # tensor([2, 3, 4, 5, 6])，不含 7

# torch.rand() — 值域 [0, 1)，不包含 1
torch.rand(3)           # 例: tensor([0.2, 0.9, 0.0])，不会出现 1.0

# random_(low, high) — [low, high)，不包含 high
torch.LongTensor(3).random_(0, 10)   # 0~9

# randperm(n) — [0, n)，不包含 n
torch.randperm(10)                    # 0~9

# Python range(start, end) — [start, end)，不包含 end
list(range(0, 5))       # [0, 1, 2, 3, 4]

# Python/Tensor 切片 [start:end] — [start, end)，不包含 end
x = torch.arange(10)    # [0,1,2,...,9]
x[2:5]                  # tensor([2, 3, 4])，不含 index 5
x[:3]                   # tensor([0, 1, 2])，不含 index 3
```

### 15.4 `nn.CrossEntropyLoss` vs `nn.NLLLoss` — 是否需要手动 softmax

```python
scores = net(inputs)   # 网络输出 raw logits

# CrossEntropyLoss: 内部自动做 softmax + log + NLLLoss
criterion = nn.CrossEntropyLoss()
loss = criterion(scores, labels)            # ✓ 直接传 raw scores

# NLLLoss: 需要自己先做 softmax + log
criterion = nn.NLLLoss()
prob = torch.softmax(scores, dim=1)
log_prob = torch.log(prob)
loss = criterion(log_prob, labels)          # ✓ 传 log-probabilities

# ⚠ 常见错误:
loss = nn.CrossEntropyLoss()(torch.softmax(scores, dim=1), labels)
# ✗ 做了两次 softmax！CrossEntropyLoss 内部会再做一次

loss = nn.NLLLoss()(scores, labels)
# ✗ NLLLoss 期望 log-probabilities，不是 raw scores
```

### 15.5 In-place 操作（带下划线 `_` 的方法）— 直接修改原张量

```python
x = torch.zeros(5)

# 非 in-place: 返回新张量，原张量不变
y = x.add(1)         # y = tensor([1,1,1,1,1])，x 仍为 tensor([0,0,0,0,0])

# In-place: 直接修改原张量，不返回新张量
x.add_(1)            # x 变为 tensor([1,1,1,1,1])
x.zero_()            # x 变为 tensor([0,0,0,0,0])
x.random_(0, 10)     # x 被填充随机整数

# ⚠ 常见 in-place 方法:
x.requires_grad_()   # 原地开启梯度追踪
x.zero_()            # 原地清零
optimizer.zero_grad() # 本质上也是对梯度做 in-place 清零
```

### 15.6 `.view()` vs `.reshape()` — 内存连续性

```python
x = torch.arange(12).view(3, 4)

# view() 要求张量在内存中连续，否则报错
y = x.t()              # 转置后内存不连续
# y.view(12)           # ✗ RuntimeError: view requires contiguous tensor

# 解决办法:
y.contiguous().view(12) # ✓ 先调用 .contiguous() 再 view
y.reshape(12)           # ✓ reshape 自动处理非连续情况（可能会复制数据）
```

### 15.7 `loss.item()` vs `loss` — 训练中记录 loss 的正确方式

```python
# ⚠ 错误: 直接累加 loss 张量，会保留整个计算图，导致内存泄漏
total_loss = 0
total_loss += loss          # ✗ loss 带有计算图，越积越多

# ✓ 正确: 用 .detach().item() 提取纯数值
total_loss = 0
total_loss += loss.detach().item()   # ✓ 提取 Python float，不保留计算图
```

### 15.8 `zero_grad()` 的位置 — 必须在 `backward()` 之前

```python
# ✓ 正确顺序: zero_grad → forward → loss → backward → step
optimizer.zero_grad()       # 1. 先清零旧梯度
scores = net(inputs)        # 2. 前向传播
loss = criterion(scores, labels)  # 3. 计算 loss
loss.backward()             # 4. 反向传播（梯度会累加到已清零的参数上）
optimizer.step()            # 5. 更新参数

# ⚠ 如果忘记 zero_grad()，梯度会在多个 batch 间累加，导致训练不稳定
```

### 15.9 损失函数对标签类型的要求

```python
# CrossEntropyLoss 和 NLLLoss 要求标签为 LongTensor（整数类别索引）
labels = torch.tensor([0, 3, 7])          # ✓ 自动推断为 LongTensor
labels = torch.LongTensor([0, 3, 7])      # ✓ 显式 LongTensor
labels = torch.Tensor([0, 3, 7])          # ✗ FloatTensor → 运行时报错

# 标签值范围: [0, num_classes)，不包含 num_classes
# 10 分类问题，标签只能是 0, 1, 2, ..., 9
```

### 15.10 `detach()` vs `torch.no_grad()` vs `requires_grad_(False)`

```python
# detach(): 从计算图中分离出一个张量，得到一个不追踪梯度的副本
y = scores.detach()     # y 与 scores 共享数据，但 y 不参与梯度计算

# torch.no_grad(): 上下文管理器，该块内所有操作不追踪梯度
with torch.no_grad():
    scores = net(inputs)  # 不构建计算图，节省内存（用于推理/评估）

# requires_grad_(False): 原地关闭某个张量的梯度追踪
x.requires_grad_(False)

# ⚠ 区别:
# - detach() 用于"摘出"某个张量去做非梯度的计算（如记录 loss、计算准确率）
# - no_grad() 用于整块推理代码，禁止所有操作构建计算图
# - requires_grad_() 用于显式控制某个张量是否追踪梯度
```

---

## 16. 速查表

| 用途 | 函数 | 示例 |
|------|------|------|
| 创建张量 | `torch.tensor`, `torch.rand`, `torch.zeros` | `torch.rand(3, 4)` |
| 查看形状 | `.size()`, `.dim()` | `x.size()` |
| 类型转换 | `.float()`, `.long()`, `.item()` | `x.float()` |
| 重塑 | `.view()` | `x.view(bs, 784)` |
| 全连接层 | `nn.Linear` | `nn.Linear(784, 10)` |
| 激活函数 | `torch.relu`, `torch.softmax` | `torch.relu(x)` |
| 损失函数 | `nn.CrossEntropyLoss`, `nn.NLLLoss` | `criterion(scores, labels)` |
| 优化器 | `torch.optim.SGD` | `SGD(net.parameters(), lr=0.01)` |
| 反向传播 | `.zero_grad()`, `.backward()`, `.step()` | 三步曲 |
| 梯度控制 | `torch.no_grad()`, `.detach()` | `with torch.no_grad():` |
| 数据读写 | `torch.load`, `torch.save` | `torch.load('data.pt')` |
| 打乱数据 | `torch.randperm` | `torch.randperm(60000)` |
| 预测类别 | `.argmax(dim=1)` | `scores.argmax(dim=1)` |

---

## 17. 边界速查

| 函数/语法 | 范围 | 包含左端 | 包含右端 | 示例 |
|-----------|------|----------|----------|------|
| `torch.arange(n)` | `[0, n)` | Yes | **No** | `arange(3)` → `[0,1,2]` |
| `torch.arange(a, b)` | `[a, b)` | Yes | **No** | `arange(2,5)` → `[2,3,4]` |
| `torch.rand()` | `[0, 1)` | Yes | **No** | 不会生成 1.0 |
| `random_(low, high)` | `[low, high)` | Yes | **No** | `random_(0,10)` → 0~9 |
| `torch.randperm(n)` | `[0, n)` | Yes | **No** | `randperm(5)` → 0~4 的排列 |
| `range(a, b)` (Python) | `[a, b)` | Yes | **No** | `range(0,3)` → 0,1,2 |
| `tensor[a:b]` 切片 | `[a, b)` | Yes | **No** | `x[1:4]` → index 1,2,3 |
| 标签值 (分类) | `[0, C)` | Yes | **No** | 10类 → 0~9 |
