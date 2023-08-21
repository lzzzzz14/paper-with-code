# self-attention

[self-attention讲解]( https://u5rpni.axshare.com/#g=14)

[self-attention代码]( https://github.com/Enzo-MiMan/cv_related_collections/blob/main/deep_learning_basic/self-attention/self_attention.py)

[b站视频讲解]([self-Attention｜自注意力机制 ｜位置编码 ｜ 理论 + 代码_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1qo4y1F7Ep/?spm_id_from=333.788&vd_source=78a547131858b1310aa0cefdfdab4b71))

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

