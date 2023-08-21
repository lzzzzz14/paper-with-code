# VOT

* 给定视频序列第一帧中目标的初始状态，这些算法跟踪剩余帧中的目标状态

## 单个对象

* VOT捕获目标序列第一帧中目标的外观特征，然后是用它来定位剩余帧中的目标



### 分类

1. 基于跟踪模型中使用的特征
   * 手工制作特征
   * 深度特征————表现明显优异
2. 基于跟踪器如何将目标对象与周围环境分开来
   * 区分性————将VOT是为二元分类任务，并将目标对象与背景分开
   * 生成性————通过搜索与每一帧中参考模版紧密匹配的最佳候选将VOT是为相似性匹配问题
3. Transformer和CNN



### Transformer和CNN

1. 不同的方式消耗图像，CNN将图像作为像素值数组，Transformer将图像分别消耗为补丁序列
2. Transformer比最先进的CNN模型更善于捕捉图像的整体信息
3. 更好的捕获图像中远程依赖关系，不会牺牲计算效率
4. 增加CNN中卷积核大小会阻碍其学习能力
5. Tranformer很难优化，需要比CNN模型更多的训练数据
6. Transformer有更多的参数，没有足够训练数据，可能导致过拟合



* 总体而言，Transformer在取代cv中的CNN，因为他的注意力机制和局部和全局特征捕获能力



## 历史

![image-20230607100829937](https://img-blog.csdnimg.cn/img_convert/1b53ae9b3d5a3509827548ce34ef0737.png)

* 早期，将CNN和Transfomer体系结合在一起，提出一种混合类型的跟踪模型
  * 将Transformer的注意力机制与CNN的分层学习能力相结合
* 后来，提出仅依赖于Transoformer架构的跟踪器



# 5个基准数据集

OTB-100、LaSOT、GOT-10k、TrackingNet、UAV123



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



