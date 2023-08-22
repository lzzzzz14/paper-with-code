## 介绍Transformer和Vision Transformer（vit）

![image-20230607104604659](https://img-blog.csdnimg.cn/img_convert/01c267f884b92cb4c1a3ae48879a6c40.png)



Transformer架构接收作为矢量序列的输入，并使用位置嵌入算法将该序列中每个标记的位置信息添加到其表示



## Self-Attention

通过计算序列中每个元素与同一序列中所有其他元素之间的注意权重来捕捉输入标记之间的语境关系

Transformer使用“Query”、“Key“、Value”抽象来计算输入序列的注意力

1. 三个不同的向量：查询向量Q、关键向量K、价值向量V

   * 通过将输入向量（x）与三个相应的矩阵相乘而产生的：W<sub>X</sub>，W<sub>K</sub>，W<sub>V</sub>（向量）
   * 所有输入向量打包在一起形成输入矩阵X，吻别产生对应的：Q、K、V
2. 通过取Q和K的点击来计算输入X得分数S（分数矩阵S）
   * 分数矩阵S提供了基于特定位置的输入向量应该对输入序列的其他部分给予多少关注


$$
S = Q \cdot K^T
$$

3. 对分数矩阵S进行归一化（得到更稳定的梯度）
   * S<sub>n</sub>是归一化的得分矩阵，d<sub>k</sub>是key向量的维度

$$
S_n = \frac{S}{\sqrt{d_k}}
$$

4. 将分数转换为概率（P）
   * 将SoftMax函数应用于归一化得分矩阵

$$
P = SoftMax(S_n)
$$

5. 得到自注意力输出值（Z）
   * 将概率（P）与值矩阵V相乘

$$
Z = P \cdot V
$$

### 自注意力统一为单个方程

$$
Attention(Q, K, V) = softmax(\frac {Q \cdot K^T} {\sqrt{d_k}}) \cdot V
$$



## Multi-Head Self-Attention

* 来提高自我关注机制的性能

* 使用了多组权重矩阵（W<sub>Q</sub>，W<sub>K</sub>，W<sub>V</sub>）-->将相同的输入数据投影到不同的子空间中
* q、k、v得每个投影版本上同时计算注意力函数以产生相应的输出值
* 最后阶段，将所有注意力头的输出连接起来，然后与另一个可训练权重矩阵W<sub>O</sub>相乘以获得多头自注意力M<sub>atten</sub>

$$
M_{atten} = Contat(head_1, ... , head_n) \cdot W_O
$$

* head<sub>i</sub>是注意力头i得输出



# vit

1. 输入图像$ I ∈ R^{H×W×C} $被分为大小相等的小块（大小为P × P）H、W、C分别代表输入图像的高度、宽度和通道数
2. 展平补丁形成大小为 $(N × (P^2 \cdot C)) $ 的2D序列（N是输入图像中提取的补丁数量）
3. 补丁嵌入位置和类信息
4. 送入编码器层，得到编码器层的输出
5. 送入MLP得到特定类别的分数

![image-20230607110633790](https://img-blog.csdnimg.cn/img_convert/5ac383ca3131c73effc7b4bf6c96823e.png)



* Swim Transformer在划分图像的非重叠窗口内执行自我注意，并引入移位窗口分区机制进行跨窗口连接（减少ViT计算复杂度）
* CVT：在ViT中加入两个基于卷积的操作，即卷积令牌嵌入和卷积投影
* VOLO引入了Outlooker轻量级注意力机制



## 