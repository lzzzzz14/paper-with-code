# Multi-Head Attention

[讲解](https://u5rpni.axshare.com/?id=4ak987&p=multi-head-attention&sc=3&g=1)

[Multi-Head Attention | 算法 + 代码_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1gV4y167rS/?spm_id_from=333.788&vd_source=78a547131858b1310aa0cefdfdab4b71)


$$
M_{atten} = Contat(head_1, ... , head_n) \cdot W_O
$$

* 感觉代码有些复杂，进行梳理

### 代码梳理

一些超参数：

* dim_in：输入中每个token的维度，即输入x的最后一个维度
* d_model：qkv总的向量长度
* num_heads：head的个数

按照前向传播的流程进行梳理，输入的x为tensor (1, 4, 2)	# (batch, 4个token, 每个token的维度)

1. 定义超参数：
   * dim_in=2，也就是输入x每个token的维度
   * d_model=6，qkv的总长度
   * num_heads=3，一共是3个头

2. 检查x的shape，分别定义为batch, n, dim_in
   * 检查这里的dim_in是否和前面定义的超参数dim_in相同

```python
batch, n, dim_in = x.shape
assert dim_in == self.dim_in
```

3. 定义nh（头的个数），dk（每个头的维度）

```python
nh = self.num_heads	# nh=3
dk = self.d_model // num_heads	# dk=2
```

4. 将x传入3个全连接层，得到q、k、v（只写了q的，k和v的操作完全一样）

* 全连接层的定义：

```python
self.linear_q = nn.Linear(dim_in, d_model)	# dim_in=2, d_model=6都是前面定义的超参数
q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
```

* x形状的变化：(1, 4, 2) &rarr; (1, 4, 6) &rarr; (1, 4, 3, 2) &rarr; (1, 3, 4, 2)
  * 首先经过一个全连接层，将最后的dim_in维度变为d_model=6	(batch, n, d_model)
  * 然后reshape，将后两个维度变为nh=3，dk=2                               (batch, n, nh, dk)
  * 最后过一个transpose，将n=4和nh=3调换位置                              (batch, nh, n, dk)

5. 做q和k的点乘，除以√dk，得到相关分数

$$
\frac{q \cdot k^T}{\sqrt{dk}}
$$

```python
self.scale = 1 / sqrt(d_model // num_heads)	# √dk
dist = torch.matmul(q, k.transpose(2, 3)) * self.scale
```

* q (1, 3, 4, 2) * k^T (1, 3, 2, 4) &rarr; S (1, 3, 4, 4) &rarr; dist (S / √dk)

6. 对dist做softmax，将相关分数转换为概率

```python
dist = torch.softmax(dist, dim=-1)	# 只对最后一个维度
```

7. 将q、k的相关性跟v进行相乘

```python
att = torch.matmil(dist, v)
att = attt.transpose(1, 2).reshape(batch, n, self.d_model)
```

* dist (1, 3, 4, 4) * v (1, 3, 4, 2) &rarr; att (1, 3, 4, 2) &rarr; transpose (1, 4, 3, 2) &rarr; reshape (1, 4, 6)

8. 最后过一个线性层（将所有head融合起来）

```python
self.fc = nn.Linear(d_model, d_model)	# 这里的d_model=6
output = self.fc(att)	
```

