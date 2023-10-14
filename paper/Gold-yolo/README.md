# Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism

### abtract

在过去的几年中，YOLO 系列模型已成为实时目标检测领域的领先方法。许多研究通过修改架构、增加数据和设计新的损失，将基线提升到更高的水平。然而，我们发现以前的模型仍然存在==信息融合问题==，尽管特征金字塔网络（FPN）和路径聚合网络（PANet）已经缓解了这个问题。因此，本研究提供了一种先进的==Gatherand-Distribute机制（GD）机制==，通过卷积和自注意力操作来实现。这种新设计的模型被命名为Gold-YOLO，它==增强了多尺度特征融合能力==，并在所有模型尺度上实现了延迟和准确性之间的理想平衡。此外，我们首次在 YOLO 系列中==实现了 MAE 式的预训练==，使 YOLO 系列模型可以从无监督预训练中受益。 Gold-YOLO-N 在 COCO val2017 数据集上获得了出色的 39.9% AP，在 T4 GPU 上获得了 1030 FPS，比之前具有类似 FPS 的 SOTA 模型 YOLOv6-3.0-N 提高了 2.4%。



## 网络结构梳理

1. Backbone提取多尺度特征图,包含B2,B3,B4,B5等。
2. Neck部分包含两个分支low-GD和high-GD:

(1) low-GD分支

- 输入:B2,B3,B4
- Low-FAM模块:用AvgPool缩放到统一尺寸
- Low-IFM模块:用可分离卷积块融合特征
- LAF模块:与相邻层进行轻量级融合
- Inject模块:注意力机制将全局信息注入当前层
- 输出:经增强的P3,P4

(2) high-GD分支

- 输入:P3,P4,P5
- High-FAM模块:AvgPool缩放到统一尺寸
- High-IFM模块:Transformer结构融合特征
- Inject模块:注意力机制融合全局信息
- 输出:增强的P4,P5

1. 最后P3,P4,P5进入头部生成检测结果。

所以低层的B2,B3,B4先进入low-GD分支;而高层的P3,P4,P5进入high-GD分支。这两个分支是并行的。

LAF模块添加在Inject之前,对局部特征进行增强。Inject模块再注入全局信息。

## method

### gold-yolo 网络框架

golo-yolo为了避免不停迭代的间接融合不同level的信息，提出一种GD(gather-and-distribute聚合-分发)的机制,整体结构如下:

![image-20231014231132546](/Users/lizhi/Library/Application Support/typora-user-images/image-20231014231132546.png)

其中GD主要分成3个模块：FAM(Feature Alignment Module，特征对齐模块)、IFM(Information Fusion Module，信息融合模块)、Inject(Information Injection Module，信息注入模块)



## low-GD模块

low-GD主要用于融合模型浅层的特征信息，此处为B2,B3,B4,B5的特征，包含上述说的2个模块，具体过程如下图所示:

![image-20231014231437126](/Users/lizhi/Library/Application Support/typora-user-images/image-20231014231437126.png)

是一个FAM模块，用下图公式表达，即先将B2,B3,B4,B5统一到B4的[h/4,w/4]的尺寸，然后在channel上concat，
$$
F_{align} = Low\_FAM([B2, B3, B4, B5])
$$

### Low-IFM模块

将FAM的输出先经过多层RepBlock提取信息，然后再按channel分成2部分，如下图公式，得到inj_p3和inj_p4，作为后续Inject的输入，此处得到的特征图作为global信息。
$$
F_{fuse} = RepBlock(F_{align}),
$$

$$
F{inj\_3}, F{inj\_4} = Split(F_{fuse})
$$

### inject模块

![image-20231014233647987](/Users/lizhi/Library/Application Support/typora-user-images/image-20231014233647987.png)

low-inject的具体过程用下图公式表达，其中global代表是用FAM模块将多层融合一起的特征，local代表的是每层的特征，比如B2
$$
F_{global\_act\_P_i} = resize(Sigmoid(Conv_{act}(F_{inj\_P_i}))),\\
F_{golbal\_embed\_P_i} = resize(Conv_{golbal\_embed\_P_i(F_{inj\_P_i})}),\\
F_{att\_fuse\_P_i} = Conv_{locan\_embed\_P_i}(B_i) * F_{global\_act\_P_i} + F_{global\_embed\_P_i},\\
P_i = RepBlock(F_{attn\_fuse\_P_i})
$$
(1) 输入:

- Flocal: 当前层的局部特征图,例如B3
- Fglobal: IFM模块输出的全局特征图

(2) 内部结构:

- 将Fglobal通过1x1卷积生成特征嵌入Fglobal_embed。
- 将Fglobal通过一个卷积生成注意力权重Fglobal_act。
- 对Fglobal_embed和Fglobal_act做上/下采样,与Flocal尺寸对齐。
- 对Flocal也做一个1x1卷积得到Flocal_embed。
- 按照Attention机制,做点积获得Ffuse = Flocal_embed * Fglobal_act + Fglobal_embed。
- 添加RepConv块进一步提炼特征。

(3) 输出:增强的特征图,例如增强后的B3 (变为了P3)。



### low-LAF模块

![image-20231014235319191](/Users/lizhi/Library/Application Support/typora-user-images/image-20231014235319191.png)

LAF是一个轻量的邻层融合模块，如下图所示，对输入的local特征(Bi或pi)先和邻域特征融合，然后再走inject模块，这样local特征图也具有了多层的信息。

(1) 输入:

- Fn-1: 前一层的特征图,如P2
- Fn:当前层特征图,如P3
- Fn+1:后一层特征图,如P4

(2) 内部结构:

- 对Fn-1和Fn+1做上/下采样,与Fn尺寸对齐。
- 将三个特征图concat在一起,1x1卷积调整通道数。
- 添加RepConv块进行特征提炼。

(3) 输出:增强后的当前层特征图Fn。



## high-GD模块

* 跟上方的low-GD模块基本相似, 只是一些模块有细微的变化, 就不再详细过一遍了

![image-20231014235740949](/Users/lizhi/Library/Application Support/typora-user-images/image-20231014235740949.png)



### high-FAM模块

$$
F_{align} = High\_FAM([P3, P4, P5]),
$$

 ### high-IFM模块

high-IFM的结构也有改动，由于high-FAM输出的特征尺寸变小了，为了充分融合全局信息，具体如下图，将low-gd中的repblock替换成了transformer模块，这里的transformer是基于卷积实现的，将linear线性层替换成了conv，layernorm替换成了batch norm。
计算公式如下：
$$
F_{fuse} = Transformer(F_{align}),\\
F_{inj\_N4}, F_{INJ\_N5} = Split(Conv1 * 1(F_{fuse})
$$

### inject模块

* 跟上面是一样的, 就不再重写了



### high-LAF模块

![image-20231015000425593](/Users/lizhi/Library/Application Support/typora-user-images/image-20231015000425593.png)

只融合两层的信息, 比low-LAF少

