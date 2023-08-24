# VOT

[arxiv](https://arxiv.org/pdf/2302.11867)

[csdn论文翻译](https://blog.csdn.net/qq_43437453/article/details/131096480)

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



## Transformer in Single Object Tracking

* 所有基于全Transformer和基于CNN-Transformer的跟踪器都使用预训练网络，并将其视为主干网络
* 一些跟踪器在跟踪期间更新他们的模型，其中一些没有更新
* 一些跟踪器使用背景信息来跟踪目标，一些没有使用

![123455](https://img-blog.csdnimg.cn/img_convert/7f1ea0f576d6ca3a7de3704abb17372a.png)

### 双流两阶段

* 使用两个相同的网络分支管道 (双流) 来提取目标图像和搜索图像的特征

* 目标模版和搜索区域的特征提取与特征融合在两个可区分的阶段中完成
* 基于CNN-Transformer都是双流双阶段

### 单流一阶段

* 使用单个网络管道, 特征提取和特征融合通过单阶段一起完成

![image-20230607162309973](https://img-blog.csdnimg.cn/img_convert/5ca973df3d334fb8fe8da0de890870f9.png)

## CNN-Transformer based Trackers

* 只基于CNN的跟踪器: 捕获目标模板和搜索区域之间的非线性交互(遮挡、变形和旋转)是不够的

1. #### [Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking](https://arxiv.org/pdf/2103.11681)

   * 一组模版补丁和搜索区域被送入一个CNN骨干网络以提取深层特征
   * 提取的模版特征被送入Transformer的**__编码器__**,利用注意力机制捕捉高质量的目标特征
   * 搜索区域的特征被送入Transformer的**__解码器__**
   * 通过前几帧的信息性目标线索与搜索区域的特征聚合在一起产生解码的特征

![image-20230607163851960](https://img-blog.csdnimg.cn/img_convert/99b162e5788e732597974809bc4570d5.png)

TrSiam: 从编码特征中裁剪目标特征, 然后与解码特征交叉相关以定位目标位置

TrDiMP: 对编码特征应用端到端判别相关滤波器 (DCF) 生成相应映射, 然后使用响应映射在搜索区域中定位目标

* 由于在此跟踪器中使用了Transformer, 一组目标模版的线索用于定位目标, 因此跟踪器能够定位具有严重外观变化的目标

2. [High-Performance Discriminative Tracking With Transformers (DTT)](https://arxiv.org/pdf/2203.13533v2)
   
   * 目标模版被送入背景场景, 然后Transformer架构捕捉到目标的最有辨识度的线索
   * 涉及进行跟踪, 不需要训练单独的判别模型 &rarr; 跟踪速度较快
3. [Transformer tracking](https://arxiv.org/pdf/2103.15436v1)
   
   三个模块: 一个CNN主干网络, 一个基于Transformer的特征融合网络, 以及一个预测网络

![image-20230607164400828](https://img-blog.csdnimg.cn/img_convert/5e3759146eac5a5f26b9bd4a636fd1b2.png)

* 目标模版和搜索区域的特征是使用ResNet50模型提取
  * 然后使用1×1卷基层进行整形

* 基于Transformer的特征融合网络有N个特征融合层 (每层有一个ECA和CFA, 分别用于增强自我注意和交叉注意)
* 最后,将融合后的特征输入预测网络,分别采用简单的分类和回归分支进行目标定位和坐标定位

4. [Learning Spatio-Temporal Transformer for Visual Tracking](https://arxiv.org/pdf/2103.17154)

   提出一种基于DETR物体检测的VOT-Transformer架构, 跟踪器被称为STARK

   * ResNet50用于提取出事目标模版、动态目标模版和搜索区域的深度特征
     * 这些特征被展平、连接,然后馈送到具有编码器和解码器架构的Transformer
   * 训练了Transfomer来捕获目标对象的**__空间和时间线索__**

![image-20230607165330774](https://img-blog.csdnimg.cn/img_convert/c2e23634949eec33c82e1032e4d7e644.png)

