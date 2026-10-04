# 目录
- [tensor&autograd](#tensorautograd)
  - [tensor张量](#tensor-张量pytorch里的多维数组可以放到-gpu加速计算autograd只能作用在张量上)
    - [1. 创建张量](#1-创建张量)
    - [2. numpy ↔ tensor](#2-numpy--tensor)
    - [3. 张量基础运算](#3-张量基础运算)
  - [autograd](#autograd)
    - [1. requires_grad](#1-requires_grad)
    - [2. backward () 反向传播求梯度](#2-backward-反向传播求梯度)
    - [3. 梯度是累积的，不是覆盖](#3-梯度是累积的不是覆盖)
    - [4. torch.no_grad () 关闭梯度追踪](#4-torchno_grad-关闭梯度追踪)
- [torch.nn 搭建神经网络](#torchnn-搭建神经网络)
  - [1. Python 类与继承](#1-python-类与继承)
  - [2. nn.Module 自定义网络](#2-nnmodule-自定义网络)
    - [nn.Linear (in_dim, out_dim) 全连接层](#nnlinear-ind-out_dim-全连接层)
    - [激活函数](#激活函数)
  - [二分类损失函数 & 优化器](#二分类损失函数--优化器)
    - [两个二分类损失对比](#两个二分类损失对比)
    - [优化器 torch.optim.Adam](#优化器-torchoptimadam)
    - [训练循环固定四步](#训练循环固定四步)
- [训练循环的4步](#训练循环的4步)
  - [1. 分步解析](#1-分步解析)
  - [2. 代码示例](#2-代码示例)
  - [3. 总结](#3-总结-zero_grad--lossbackward--optimizerstep)
- [区分训练模式 model.train() 和评估模式 model.eval()](#区分训练模式-modeltrain和评估模式-modeleval)
  - [1. what](#1-what)
    - [model.train() 训练模式](#modeltrain-训练模式)
    - [model.eval() 评估/推理模式](#modeleval-评估推理模式inference)
  - [2. 关键配套：with torch.no_grad()](#2-关键配套with-torchno_grad)
  - [3. 完整阶段4代码](#3-完整阶段4代码)
  - [4. 必背规范清单](#4-必背规范清单)
- [加入 BatchNorm、Dropout](#加入-batchnormdropout直观看懂二者作用)
  - [层放置标准规则](#层放置标准规则)
  - [修改后的网络代码](#修改后的网络代码)
  - [重点理解](#重点理解)
    - [nn.Dropout(p=0.3)](#1-nndropoutp03)
    - [nn.BatchNorm1d(8)](#2-nnbatchnorm1d8)
  - [可能易错](#可能易错)
  - [一些思考](#一些思考)
- [Dataset & DataLoader 数据集封装](#dataset--dataloader-数据集封装)
  - [Dataset 类](#dataset-类)
    - [最简 Demo](#最简-demo)
  - [DataLoader](#32-dataloader)
  - [遍历 DataLoader](#遍历-dataloader)
- [泰坦尼克数据预处理](#泰坦尼克数据预处理)
  - [读取csv、字段筛选、缺失值处理](#读取csv字段筛选缺失值处理)
    - [泰坦尼克列](#泰坦尼克列)
    - [删除没用的列](#删除没用的列直接丢掉不参与模型)
    - [缺失值填充规则](#缺失值填充规则)
  - [类别特征编码](#42-类别特征编码)
  - [特征标准化](#43-特征标准化)
  - [划分训练集/测试集，固定随机种子](#划分训练集测试集固定随机种子)
  - [完整Demo代码](#完整demo代码)

# tensor&autograd
## tensor
张量（Tensor）就是 PyTorch 里的多维数组，可以放到 GPU 加速计算，**Autograd 只能作用在张量上**。
### 1. 创建张量

```python
import torch

# 1. 直接从列表创建张量
t1 = torch.tensor([1.0, 2.0, 3.0])
print(t1)
print(t1.shape)   # 查看形状
print(t1.dtype)   # 查看数据类型
```

- 重点：**做自动微分，必须是浮点型 `float32 / float64`，整型不能求梯度！**
   - 梯度本质是函数的导数，导数描述的是 “微小变化” 带来的函数值变化，需要连续的数值范围。
   - 而整型是离散的，比如从 1 变到 2 是 “跳变”，没有 “微小变化” 的概念，所以 Autograd 不支持对整型张量求梯度。

### 2. numpy ↔ tensor

我们用 pandas 处理完数据得到 numpy 数组，最后要转成 tensor 喂进网络

```python
import numpy as np

# numpy数组 -> tensor
arr = np.array([[1,2],[3,4]], dtype=np.float32)
t = torch.from_numpy(arr)
print(t)

# tensor -> numpy
arr_back = t.numpy()
print(arr_back)
```


> 如果张量开启了梯度追踪（requires_grad=True），直接调用`.numpy()`会报错，必须先用`.detach()`

### 3. 张量基础运算

**只要运算里包含一个 requires_grad=True 的张量，运算结果会自动带上梯度追踪**

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)
c = a * b
print(c.requires_grad) # True，继承了a的梯度追踪
```

## autograd
> **前向计算时记录每一步数学运算，形成计算图；调用 backward ()，从最终输出（loss）沿着图反向链式法则求导，算出每个叶子张量的梯度**

### 1. requires_grad

- `requires_grad=False`（默认）：不记录计算图，不计算梯度，推理阶段用
- `requires_grad=True`：记录运算，后续可以求导（网络权重、输入标签不用，模型参数需要）

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0)
y.requires_grad_(True) # 原地开启梯度追踪，下划线=原地in-place操作

z = x**2 + 3*y
print(z)
print(z.grad_fn) # grad_fn：记录生成这个张量的运算（计算图节点）
```

运行输出会看到 `AddBackward0`，代表 z 是加法运算得到的张量。

### 2. backward () 反向传播求梯度

当最终结果是**标量**（损失 loss 就是标量！），直接调用`.backward()`

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
# 前向计算
z = x**2 + 3 * y
# 反向传播，求dz/dx，dz/dy
z.backward()

print(x.grad) # dz/dx = 2x = 4
print(y.grad) # dz/dy = 3
```

执行完`z.backward()`，自动求导结果存入`.grad`属性。

### 3. 梯度是累积的，不是覆盖

每次调用`backward()`，新梯度会**叠加**到原来的`.grad`，不会自动清空。

```python
import torch
x = torch.tensor(2.0, requires_grad=True)

loss = x ** 2
loss.backward()
print(x.grad) # 4

# 再次计算，不清零！
loss = x ** 2
loss.backward()
print(x.grad) # 8！4+4，累加，不是重置为4
```
- 同一个张量的.grad 属性是 “累加存储” 的，不管你用它算哪个函数的梯度，新结果都会加上旧结果
  - 解决方法：`x.grad.zero_()` 原地清零。(`zero_grad()` 放在**loss 计算之后、backward 之前**)
- 为什么要这么计算呢？大 batch 的梯度，数学上等于这个大 batch 里所有单个样本梯度的总和，再除以样本数。
     - 如果显存不够放整个大 batch，就拆成 N 个小 batch，每个小 batch 算完梯度后不清零，累加起来，等 N 个小 batch 都算完，总梯度就是所有样本梯度的总和，再除以总样本数

在完整神经网络里，等价于 `optimizer.zero_grad()`，**每一轮训练开头必须执行**。

### 4. torch.no_grad () 关闭梯度追踪

推理、评估阶段不需要更新权重，不需要计算梯度。
使用`with torch.no_grad():`，上下文内所有运算**不构建计算图，节省内存、提速**。

```python
import torch
x = torch.tensor(3.0, requires_grad=True)

with torch.no_grad():
    y = x * 5
    print(y.requires_grad) # False，不追踪梯度

# 退出上下文，恢复追踪
z = x * 5
print(z.requires_grad) # True
```


# torch.nn 搭建神经网络
## 1. Python 类与继承

`nn.Module` 是 PyTorch 自带的**父类**，写自己的网络时，要写一个类去继承它

- `__init__`：构造函数，**只放网络层定义**（权重在这里创建），不写计算过程
- `forward(self, x)`：前向传播函数，**写数据流动、运算逻辑**，输入 x，返回预测输出
- `self`：代表当前这个网络实例本身

> 
> 一句话区分：
> `__init__`：**定义有哪些层**（买好积木）
> `forward`：**数据怎么经过这些层**（搭积木）

## 2. nn.Module 自定义网络

### nn.Linear (in_dim, out_dim) 全连接层

`in_dim`：输入特征数量；`out_dim`：输出特征数量
内部自动创建权重 w、偏置 b，**这些参数默认自带 requires_grad=True**，不用手动设置

```python
import torch
import torch.nn as nn

# 自定义网络，继承 nn.Module
class TwoLayerNet(nn.Module):
    def __init__(self, in_feature):
        # 必须调用父类的初始化！super()
        super().__init__()
        # 定义网络层
        self.fc1 = nn.Linear(in_feature, 16) # 第一层：输入in_feature个特征，输出16
        self.fc2 = nn.Linear(16, 1)          # 第二层：输入16，输出1（二分类）

    def forward(self, x):
        # 前向计算：数据x在这里流动
        x = self.fc1(x)
        x = torch.relu(x) # ReLU激活函数，引入非线性
        x = self.fc2(x)
        return x

# 实例化网络：假设输入是5维特征
model = TwoLayerNet(in_feature=5)  

# 造一批假数据：batch_size=4，每个样本5个特征
x = torch.randn(4,5)
pred = model(x) # 直接调用model(x)，自动执行forward，不要手动调用forward！
print(pred.shape)
```

> 
> 重点：不要写 `model.forward(x)`，直接 `model(x)` 就行，PyTorch 内部会额外处理钩子等逻辑。
> 定义`self.fc1 = nn.Linear(4,8)`，就是告诉网络 “我有一个叫 fc1 的全连接层，输入 4 维特征，输出 8 维”
> 输出的 8 维是 8 个 “中间特征”，可以理解为模型对原始 4 个特征的 “加工后的新特征”
> `x = torch.relu(x)` 是对 fc1 的输出做 ReLU 激活，把负数变成 0，这样能让模型学习更复杂的非线性关系，避免多层网络退化成单层效果

### 激活函数

- `ReLU`：把负数置 0，增加非线性。如果没有非线性，再多全连接层等价于一层，模型没有拟合能力。
- `Sigmoid`：把任意实数压缩到 (0,1)，用来表示概率。

> 中间层加 ReLU 是为了过滤无效信息、增加非线性，让模型能学到更复杂的特征组合。
> 比如中间层某个神经元输出负数，可能代表这个特征组合对当前任务没用，ReLU 把它变成 0，相当于 “忽略这个无效组合”。
> 重要：`BCEWithLogitsLoss` 会内部自带 sigmoid，**网络最后一层不要手动加 sigmoid**！
> 最后一层后不加ReLU激活：BCEWithLogitsLoss 需要网络输出原始的 logits，加了 ReLU 会截断负数，导致损失计算不准。

## 二分类损失函数 & 优化器

### 两个二分类损失对比

1. `nn.BCEWithLogitsLoss()` **优先推荐**
   - 输入：网络原始输出（logits，**不加 sigmoid**） + 标签（0/1）
   - 内部自动做 sigmoid，数值更稳定，不容易梯度爆炸。
2. `nn.BCELoss()`
   - 输入：必须是 0~1 之间的概率，网络输出需要手动 sigmoid。
   - 容易出现数值不稳定，项目尽量不用。

### 优化器 torch.optim.Adam

优化器的作用：拿到模型所有参数，根据梯度更新权重，降低 loss。

```python
# 实例化优化器：传入模型参数 + 学习率lr
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 训练循环固定四步

```python
# 1. 梯度清零
optimizer.zero_grad()
# 2. 前向传播，计算预测值、损失
pred = model(x)
loss = loss_fn(pred, label)
# 3. 反向传播，计算所有参数梯度
loss.backward()
# 4. 优化器更新权重
optimizer.step()
```
```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,in_features):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=1)

    def forward(self,x):
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.fc2(x)
        return x

model=Net(in_features=4)

# ========== 新增部分 ==========
# 1. 损失函数（BCEWithLogitsLoss，对应二分类）
loss_fn = nn.BCEWithLogitsLoss()
# 2. 优化器，管理模型里所有fc1、fc2的权重
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 造假标签：3个样本，二分类标签0/1
y_true = torch.tensor([[1.0],[0.0],[1.0]])

# 训练循环！反复更新权重
for epoch in range(100):
    # 前向传播
    pred = model(x)
    # 计算损失
    loss = loss_fn(pred, y_true)

    # 反向传播三部曲
    optimizer.zero_grad() # 清空上一轮梯度
    loss.backward()       # 反向求梯度
    optimizer.step()      # 优化器更新权重

    if epoch %10 ==0:
        print(f"epoch {epoch}, loss: {loss.item():.4f}")


```
#  训练循环的4步
> 完整训练一轮的固定顺序：
> **前向传播 → 计算loss → 清空梯度zero_grad → 反向求导backward → 更新权重step**

## 1. 分步解析
### ① `optimizer.zero_grad()` 清空梯度
- 网络权重上会保存上一次算出来的梯度。
- PyTorch**默认梯度是累加**，不是覆盖。
- 如果不清空，新梯度会叠加到老梯度上，计算就错了。
> **每次迭代开始前，把上一轮残留梯度清零**。

### ② `loss.backward()` 反向传播，求梯度
- 从loss往回，沿着网络链式法则，自动求导。
- 算出：**每一个权重（fc1、fc2里全部w和b）对loss的偏导（梯度）**
- 梯度存在`参数.grad`里面，只算梯度，**不修改权重**！
> backward只负责算梯度，不改参数。

### ③ `optimizer.step()` 更新权重
- 优化器拿到刚刚backward算出的梯度，按照学习率lr，更新每一层权重。
公式（SGD）：
$$w_{new}=w_{old} - lr \times \nabla w$$
> 这一步**才真正修改fc1、fc2的权重**，也就是模型在“学习”。

---

## 2. 代码示例
```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,in_features):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=1)

    def forward(self,x):
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.fc2(x)
        return x

# 实例模型
model=Net(in_features=4)

# 损失函数：二分类，接收logits
loss_fn = nn.BCEWithLogitsLoss()
# 优化器，传入模型全部参数，学习率lr=0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 数据
x = torch.randn(3,4)
y_true = torch.tensor([[1.0],[0.0],[1.0]])

# 训练循环
for epoch in range(100):
    # 1. 前向传播
    pred = model(x)
    # 2. 计算损失
    loss = loss_fn(pred, y_true)

    # -------- 反向传播+更新权重 --------
    optimizer.zero_grad() # 清空历史梯度
    loss.backward()       # 反向求梯度
    optimizer.step()      # 更新权重

    if epoch % 10 ==0:
        print(f"epoch {epoch:3d} | loss: {loss.item():.4f}")
```
运行后，你会看到loss慢慢下降，代表模型预测越来越贴近标签。

## 3. 总结
`zero_grad()` → `loss.backward()` → `optimizer.step()`


# 区分**训练模式 model.train()**和**评估模式 model.eval()**

模型有两种工作状态，**网络的行为会跟着状态切换**，不是单纯跑不跑循环的区别。
> 重点：这对Dropout、BatchNorm层生效；简单网络（只有Linear+ReLU）虽然看不出效果，但这是写工程代码的硬性规范，必须养成习惯。

## 1. what
### `model.train()` 训练模式
- 告诉模型：**现在正在训练，要开启训练专属功能**
  - Dropout：随机关掉一部分神经元
     - 训练时随机关神经元，是**每个样本前向传播时都随机关一次**，让模型不依赖某个 “关键神经元”，强迫每个神经元都学习有用特征，有效防止过拟合。
  - BatchNorm：用当前batch的均值方差，更新滑动均值
  - Dropout 层通常加在全连接层或卷积层之后、激活函数之前 / 之后，用来防止过拟合，不能随便插在输入层之前或输出层之后。
  - BatchNorm 层一般放在卷积层或全连接层之后、激活函数之前，目的是稳定中间输出分布，加速训练
- 计算图会构建，**可以做反向传播、算梯度、更新权重**
- 放在**训练循环最开头**

### `model.eval()` 评估/推理模式（inference）
- 告诉模型：**现在是预测，不再学习，关闭训练专属功能**
  - Dropout：全部神经元启用，不再随机丢弃
  - BatchNorm：使用已经学习好的全局均值方差，不再更新
- **不计算梯度**（推荐搭配`with torch.no_grad():`）
- 用于：验证集测试、模型上线预测，**不更新权重**

> 两层全连接网络，没有Dropout/BatchNorm，`train()` / `eval()` 好像没差别，但以后网络复杂了，不加就会出大bug！

## 2. 关键配套：`with torch.no_grad():`
即使写了`model.eval()`，PyTorch默认依然会记录计算图，占用显存。
`torch.no_grad()` 的作用：**关闭梯度记录，不构建计算图，省显存、提速**
> 评估预测时，不需要反向传播，完全不需要保存计算图。

## 3. 完整阶段4代码
```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,in_features):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=1)

    def forward(self,x):
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.fc2(x)
        return x

model=Net(in_features=4)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练数据
x_train = torch.randn(3,4)#3个样本4个特征
y_train = torch.tensor([[1.0],[0.0],[1.0]])#3个样本各自的结果
# 验证数据（模拟 unseen 的新数据）
x_val = torch.randn(2,4)
y_val = torch.tensor([[0.0],[1.0]])

# ============ 训练循环 ============
for epoch in range(100):
    # 切换为训练模式
    model.train()

    pred = model(x_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每20轮，做一次验证
    if epoch %20 ==0:
        # 切换评估模式
        model.eval()
        # 关闭梯度记录
        with torch.no_grad():
            pred_val = model(x_val)
            loss_val = loss_fn(pred_val, y_val)
        print(f"epoch {epoch:3d} | train_loss:{loss.item():.4f} | val_loss:{loss_val.item():.4f}")
```

## 4. 必背规范清单
1. 训练阶段：`model.train()`
2. 验证/预测阶段：`model.eval()` + `with torch.no_grad():`
3. 切换模式只改变**层的行为**，不会修改权重本身
4. `with torch.no_grad()` 只影响梯度记录，**不改变权重**

神经网络的训练逻辑本质就是 “猜答案→看猜得对不对→调整怎么猜→再猜” 的循环


# 加入 BatchNorm、Dropout

> 1d：一维，用于全连接网络；卷积网络用BatchNorm2d

## 层放置标准规则
1. **nn.Linear → nn.BatchNorm1d → 激活ReLU → nn.Dropout**
- Linear：做线性变换
- BatchNorm1d：标准化这一层的输出，稳住分布（放在激活前面）
- ReLU：非线性激活
- Dropout：随机屏蔽神经元（放在激活之后）
> 不要放在输出层后面！输出层不能加Dropout/BN

### 修改后的网络代码
```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,in_features):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=8)
        # BatchNorm：输入通道数=上一层输出维度=8
        self.bn1 = nn.BatchNorm1d(num_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=1)
        self.dropout = nn.Dropout(p=0.3) # p=0.3：训练时30%神经元被随机关掉

    def forward(self,x):
        x=self.fc1(x)
        x=self.bn1(x)    # BN：标准化
        x=torch.relu(x)
        x=self.dropout(x)# Dropout：随机屏蔽（只在train模式生效）
        x=self.fc2(x)
        return x

model=Net(in_features=4)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 数据
x_train = torch.randn(3,4)
y_train = torch.tensor([[1.0],[0.0],[1.0]])
x_val = torch.randn(2,4)
y_val = torch.tensor([[0.0],[1.0]])

# ============ 训练循环 ============
for epoch in range(100):
    model.train() # 开启Dropout随机关闭 + BN使用batch均值方差
    pred = model(x_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch %20 ==0:
        model.eval() # 关闭Dropout + BN使用滑动平均全局统计量
        with torch.no_grad():
            pred_val = model(x_val)
            loss_val = loss_fn(pred_val, y_val)
        print(f"epoch {epoch:3d} | train_loss:{loss.item():.4f} | val_loss:{loss_val.item():.4f}")
```

## 重点理解
### 1. `nn.Dropout(p=0.3)`
- `model.train()`：每一次前向，随机关掉30%神经元。**每一轮、每个样本随机选择**，强制模型不能依赖少数神经元，防过拟合。
- `model.eval()`：Dropout直接失效，**全部神经元启用**。

> p是丢弃概率，一般取0.2~0.5，不是越大越好，p太高会欠拟合。

### 2. `nn.BatchNorm1d(8)`
参数`num_features=8`，**必须等于上一层输出的神经元数量**！
- `model.train()`：用当前batch的均值、方差做标准化；同时更新滑动平均（全局统计量）
- `model.eval()`：不再计算当前batch均值方差，直接使用训练阶段累积好的**全局滑动均值方差**

> 这就是为什么BN需要区分train/eval模式！之前只有Linear的时候感受不到差异。

## 可能易错
1. BN的`num_features`必须和上一层输出维度匹配
- BatchNorm 的 num_features 必须和上一层输出维度匹配，每层输出维度不同，所以每层 BN 都要单独定义
2. Dropout**不要放在最后输出层**：输出不能随机丢信息
3. BN一般放在**Linear之后、激活之前**，是工程上的常规写法

## 一些思考
1.why 这个顺序：
- BatchNorm 的作用是稳定线性层输出的分布，让数据落在激活函数的敏感区域，这时候再用 ReLU 激活，能更好地保留特征；
- 如果先激活再 BN，ReLU 可能会把很多值变成 0，导致 BN 计算的均值方差不准，失去稳定分布的意义。
- 而 Dropout 放在激活之后，是因为激活后的特征更有区分度，随机屏蔽这些特征能强迫模型学习更鲁棒的模式；
- 如果放在激活前，线性输出还没经过非线性变换，屏蔽后对特征的影响没那么有效。
2. dropout的到底是什么
Dropout 是直接对上一层输出的特征进行随机 “切断”，让部分特征无法传到下一层，而不是控制下一层是否接收。
所以放在激活后，切断的是 “经过筛选的有用特征”，能逼着模型用其他特征也学会预测，这样泛化能力更强。
如果放在激活前，切断的是 “还没筛选的原始线性特征”，对模型强迫学习冗余特征的作用就弱一些。
3. batchnorm
在训练神经网络时，每一层的输入数据分布会随着前一层参数更新而变化，这种现象叫 “内部协变量偏移”，
会让后一层需要不断适应新分布，训练变慢且不稳定。BatchNorm 的原理是，对每一层的输入数据，在每个 batch 内计算均值和方差，然后把数据标准化成均值 0、方差 1 的分布，再通过两个可学习的参数缩放和平移，恢复数据的表达能力。这样一来，每层输入分布被 “拉回” 稳定范围，后一层就不用频繁适应，训练速度加快，模型也更鲁棒。
比如训练图像分类模型时，加入 BatchNorm 后，即使学习率调大一点，也不容易出现梯度爆炸或消失。

# Dataset & DataLoader 数据集封装

## Dataset 类

Dataset 是一个抽象父类，继承它必须**强制实现两个魔法函数**

1. `__len__(self)`：返回整个数据集一共有多少样本，`len(数据集实例)`会自动调用它
2. `__getitem__(self, idx)`：按索引`idx`，取出**单条样本**：返回（特征张量，标签张量）

> 
> 重点：`idx`是单个样本下标，不是 batch！这里一次只拿一行数据。

### 最简 Demo

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 自定义数据集类
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        # X：numpy特征数组 [样本数,特征数]
        # y：numpy标签数组 [样本数]
        self.X = X
        self.y = y

    def __len__(self):
        # 返回样本总数
        return len(self.X)

    def __getitem__(self, idx):
        # 根据idx取出第idx条样本
        feat = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return feat, label
```

## 3.2 DataLoader

DataLoader 用来包装上面的 Dataset，自动做这些事：

- 按`batch_size`，把多条样本打包成一个批次
- `shuffle=True`：每个 epoch 开始，打乱全部数据（**训练集一定要开 shuffle；验证集不要 shuffle**）
    - 验证集 shuffle，每次评估的样本顺序不一样，计算出的 loss 和 acc 就会有波动，无法准确判断模型是否真的在变好。
    - 而训练集 shuffle 是为了让模型每次看到不同的样本组合，避免过拟合到固定顺序的样本
- 自动堆叠张量，把一批样本拼成`[batch_size, feature_num]`的形状，直接喂给网络

参数重点：

```python
# 构造DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,      # 一批4个乘客
    shuffle=True,      # 训练集打乱
    drop_last=False    # 最后不足batch_size的样本要不要丢掉
)
```

### 遍历 DataLoader

```python
# 遍历每个batch
for batch_x, batch_y in train_loader:
    print("batch_x shape:", batch_x.shape) # [4,特征数]
    print("batch_y shape:", batch_y.shape) # [4]
    break
```

> 
> 易错点：
> Dataset 拿**一条**；DataLoader 迭代出来是**一整个 batch**。
> 不要在`__getitem__`里面写 batch 相关逻辑！


# 泰坦尼克数据预处理
> 前提：阶段1/2/3全部掌握，不再重复张量、Dataset、DataLoader。
> 核心铁律：**所有预处理统计量（均值、方差）只能从训练集计算，测试集直接复用！绝对不能用测试集的均值方差**
目标：Pandas清洗train.csv，得到numpy数组，转张量送入Dataset。

## 读取csv、字段筛选、缺失值处理
### 泰坦尼克列
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

### 删除没用的列（直接丢掉，不参与模型）
- PassengerId：只是编号，无预测信息
- Name：名字，文本，我们不做NLP
- Ticket：票号，杂乱字符串，没用
- Cabin：大量缺失，信息太少，直接删掉

保留：`Survived(标签), Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`

### 缺失值填充规则
1. **Age（年龄）**：有缺失。用**训练集Age的平均值填充**（不要全局全部数据均值！）
2. **Embarked（登船港口）**：只有很少缺失，用训练集最多的类别填充
3. Cabin：缺失率极高，直接删整列

> 重点：先划分训练/测试集，再计算均值用来填充！顺序不能反过来。
> 错误：先填充全部数据，再划分训练测试（泄露测试信息）
> 正确：先切分，拿训练集的统计值，填充训练集和测试集。

## 4.2 类别特征编码
两个类别特征：
1. Sex：male / female → 二分类，简单映射 male=0，female=1
2. Embarked：C/Q/S 三个港口 → 独热编码（OneHot），会新增3列

> 独热编码举例：
> S → [1,0,0]
> C → [0,1,0]
> Q → [0,0,1]

## 4.3 特征标准化
只对**连续数值特征**做标准化：Age，Fare
公式：
$$x_{scaled} = \frac{x-\mu_{train}}{\sigma_{train}}$$
$\mu_{train}$ = 训练集该特征均值
$\sigma_{train}$ = 训练集该特征标准差

严禁：测试集自己算μ、σ。测试集必须**直接使用训练集算好的均值方差**。
> 原因：测试集是模拟未来看不见的数据，真实场景你拿不到测试集分布，如果用测试集统计量，就是**数据泄露**

离散特征（Pclass、Sex、Embarked独热）**不用标准化**。

## 划分训练集/测试集，固定随机种子
用`sklearn.model_selection.train_test_split`
- test_size=0.2：20%做验证集
- random_state=42 固定种子，保证每次划分结果一模一样，方便复现
- stratify=y：分层划分，保证训练集、验证集里面幸存/遇难比例和原始数据一致（二分类表格数据推荐）

## 完整Demo代码
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1.读入数据
df = pd.read_csv("train.csv")

# 删掉不需要的列
drop_cols = ["PassengerId","Name","Ticket","Cabin"]
df = df.drop(columns=drop_cols)

# 2.划分标签y和特征X
y = df["Survived"].values
X_df = df.drop(columns=["Survived"])

# 先分割！！！【核心顺序：先分割，再做所有统计计算，防止数据泄露】
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_df, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 从训练集拿到统计信息 ----------
# Age填充均值
age_mean = X_train_df["Age"].mean()
# Embarked取最多的类别
embark_most = X_train_df["Embarked"].mode()[0]

# 定义一个预处理函数：训练集/验证集都调用，但是统计量来自训练集
def preprocess_df(data_df, age_mean_fill, embark_fill):
    df_tmp = data_df.copy()
    # 缺失值填充
    df_tmp["Age"] = df_tmp["Age"].fillna(age_mean_fill)
    df_tmp["Embarked"] = df_tmp["Embarked"].fillna(embark_fill)
    
    # Sex编码
    df_tmp["Sex"] = df_tmp["Sex"].map({"male":0, "female":1})
    
    # Embarked独热编码
    ohe = OneHotEncoder(sparse_output=False, drop="first")
    embark_ohe = ohe.fit_transform(df_tmp[["Embarked"]])
    embark_df = pd.DataFrame(embark_ohe, columns=["Embark_C","Embark_Q"])
    df_tmp = pd.concat([df_tmp, embark_df], axis=1)
    df_tmp = df_tmp.drop("Embarked", axis=1)
    return df_tmp

# 分别预处理训练集、验证集
X_train_processed = preprocess_df(X_train_df, age_mean, embark_most)
X_val_processed = preprocess_df(X_val_df, age_mean, embark_most)

# 连续特征标准化（Age,Fare）
age_mu = X_train_processed["Age"].mean()
age_std = X_train_processed["Age"].std()
fare_mu = X_train_processed["Fare"].mean()
fare_std = X_train_processed["Fare"].std()

def standardize(df, mu_age, std_age, mu_fare, std_fare):
    df_tmp = df.copy()
    df_tmp["Age"] = (df_tmp["Age"] - mu_age)/std_age
    df_tmp["Fare"] = (df_tmp["Fare"] - mu_fare)/std_fare
    return df_tmp

X_train_final = standardize(X_train_processed, age_mu, age_std, fare_mu, fare_std)
X_val_final = standardize(X_val_processed, age_mu, age_std, fare_mu, fare_std)

# 转numpy数组
X_train = X_train_final.values
X_val = X_val_final.values
```
