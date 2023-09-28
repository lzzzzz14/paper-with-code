# EfficientViT

## 创新点

* ReLU-based Attention，light MSA



EfficientViT是一种用于边缘设备上的语义分割任务的新型模型。它的核心特点包括：

1. **轻量级多尺度注意力模块：** EfficientViT引入了一种创新的多尺度注意力模块，该模块基于基于ReLU的全局注意力。这个模块可以在保持硬件效率的同时实现全局感受野和多尺度学习，这对于语义分割任务非常重要。
2. **硬件高效性：** 与先前的语义分割模型不同，EfficientViT的设计考虑了在边缘设备上的硬件效率。它避免了计算复杂度高的操作，如二次复杂度的自注意力和大卷积核，从而在移动设备上表现更出色。
3. **性能提升：** 在流行的语义分割基准数据集上，EfficientViT相对于先前的最先进语义分割模型取得了显著的性能提升。同时，它不涉及硬件低效操作，因此FLOPs减少可以轻松转化为移动设备上的延迟减少。
4. **速度提升：** 在移动设备上，EfficientViT相对于先前的模型表现出了显著的速度提升，例如在高通骁龙8Gen1 CPU上，它可以比其他模型快数倍，而且在不损失准确性的情况下。



通常的自注意力可以被写成这样
$$
O_i = \sum_{j=1}^{N}\frac{Sim(Q_i,K_j)}{\sum_{j=1}^{N}Sim(Q_i,K_j)}V_j
$$

1. 输入向量x，有N个元素（向量的长度），每个元素有f个特征（每个元素的特征维度）

2. 通过三个科学系的线性投影矩阵W~Q~，W~K~，W~V~，这些矩阵的维度：f×d

   将x投影到三个不同的空间Q、K、V &rarr; 形状（N，d）

3. 使用相似度函数Sim(Q, K)来度量Q和K之间的相关性（在自注意力机制中用于权重计算）

$$
Sim(Q,K) = exp(\frac{QK^T}{\sqrt d})
$$

4. 通过相似度加权的方式，得到每个输出元素Oi

* 这是一种用于计算全局感受野的自注意力机制，将输入向量映射到不同的空间，然后通过相似度函数来计算权重，最终生成全局感受野的输出。



## lightweight MSA (Multi-Scale Attention)

## Lightwight ReLU-based Attention

* 在基于ReLU的全局注意力中，相似度函数被定义为：

$$
Sim(Q, k) = ReLU(Q)ReLU(K)^T
$$

* 输出可以改写为：

$$
O_i = \sum_{j=1}^{N}\frac{ReLU(Q_i)ReLU(K_j)^T}{\sum_{j=1}^{N}\textcolor{red}{ReLU(Q_i)}ReLU(K_j)^T}V_j\\
=\frac{\sum_{j=1}^{N}(ReLU(Q_i)ReLU(K_j)^T)V_j}{\textcolor{red}{ReLU(Q_i)}\sum_{j=1}^{N}ReLU(K_j)^T}
$$

* 利用矩阵乘法的结合性质，将计算复杂性和内存占用从二次降低到线性，而不改变功能

$$
O_i=\frac{\sum_{j=1}^{N}\textcolor{red}{[ReLU(Q_i)ReLU(K_j)^T]}V_j}{ReLU(Q_i)\sum_{j=1}^{N}ReLU(K_j)^T}\\
=\frac{\sum_{j=1}^{N}ReLU(Q_i)\textcolor{red}{[(ReLU(K_j)^TV_j)]}}{ReLU(Q_i)\sum_{j=1}^{N}ReLU(K_j)^T}\\
=\frac{\textcolor{blue}{ReLU(Q_i)}(\sum_{j=1}^{N}{ReLU(K_j)^TV_j)}}{ReLU(Q_i)\sum_{j=1}^{N}ReLU(K_j)^T}
$$

* 我们只需要计算一次$(\sum_{j=1}^{N}ReLU(K_j)^TV_j\in R^{d×d})$ 和 $(\sum_{j=1}^{N}ReLU(K_j)^T\in R^{d×1})$，然后可以为每个query（Q）重复使用它们，因此只需要O(N)的计算成本和O(N)的内存
* ReLU的全局注意力，不设计softmax这种对硬件不友好的操作，因此在硬件上的执行会更高效，更适合在硬件上执行

## Generate Multi-Scale Tokends

为了更好地捕获全局信息，我们采用了卷积操作来聚合查询（Q）、键（K）和值（V）令牌的信息。这个聚合操作的目的是获取多尺度的信息，以帮助模型更好地理解输入数据。

![image-20230927105151369](C:\Users\12570\AppData\Roaming\Typora\typora-user-images\image-20230927105151369.png)

为了提高计算效率，我们将卷积操作转化为了组卷积，这意味着我们将卷积操作中的计算任务分组进行，以减少总体计算量。同时，对查询、键和值令牌分别进行了相同的组卷积操作，以确保每个部分都能从多尺度信息中受益。

![image-20230927105216719](C:\Users\12570\AppData\Roaming\Typora\typora-user-images\image-20230927105216719.png)

