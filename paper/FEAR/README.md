# FEAR: Fast, Efficient, Accurate and Robust Visual Tracker

[[2112.07957\] FEAR: Fast, Efficient, Accurate and Robust Visual Tracker (arxiv.org)](https://arxiv.org/abs/2112.07957)

[FEARTracker: Official repo for FEAR: Fast, Efficient, Accurate and Robust Visual Tracker (ECCV 2022) (github.com)](https://github.com/PinataFarms/FEARTracker)



* Siamese visual trackers

### 结构图

![image-20231009145513718](/Users/lizhi/Library/Application Support/typora-user-images/image-20231009145513718.png)

1. 特征提取
2. 双模板表示
3. 像素级融合模块
4. bounding box
5. 分类头



## method

* static template: I~T~
* Search image crop: I~S~
* Dynamic template: I~d~

模版特征融合: 计算 I~T~ feature 和 I~d~ feature 的线性插值 (linear interpolation)

融合后的特征与I~S~ feature 进行像素级别融合 (pixel-wise fusion)

然后传给分类和回归的子网络



### 特征提取网络

backbone的输出应该具有足够高的分辨率, 这有利于提取位置信息; 但分辨率太高会增加后续层的计算量

作者发现, 保持图像原有的分辨率是最佳的平衡

* FEAR-M —— ResNet-50
* FEAR-L —— RegNet

这里的backbone作者采用前四层在ImageNet上进行预训练, 过完四阶段之后特征图的长宽比原图缩小了16倍

* FEAR-XS —— FBNet

为了保证输出的特征图有相同的通道, 使用了一个 AdjustLayer(CNN + BN) 来改变他们的通道数

由于预测头非常复杂, 所以轻量化的encoder并不能提升模型的效率, 需要设计轻量化和准确的decoder



### 双模板表示

动态模版的更新允许模型在推理期间捕捉目标的外观变化并且不需要实时进行优化

![image-20231009165521363](/Users/lizhi/Library/Application Support/typora-user-images/image-20231009165521363.png)

* main static template: I~T~
* search image: I~S~
* dynamic template image: I~d~

* F~d~: I~d~ 的 feature map
* F~T~: I~T~ 的 feature map
* F~S~: I~S~ 的 feature map

通过一个可学习参数w, 来计算 F~d~ 和 F~T~ 的线性插值:
$$
F'_T = (1-w)F_T + wF_d
$$
然后计算 F~T~‘ 和 F~S~ 的embedding

* e~S~: 用分类置信度分数对特征图 F~S~ 进行加权平均池化 (WAP)
* e~T~: 对 F~T~‘ 进行平均池化

计算 e~s~ 和 e~T~ 的余弦相似度

在推理中, 每过N帧, 我们选择和搜索图片余弦相似度最高的双模版表示, 然后通过预测框来更新动态模版

* I~N~: negative crop

同时, 我们从不包含目标物体的帧当中选取 I~N~ 作为负样本

* e~N~: 把 I~N~ 过一遍特征提取网络, 在通过WAP (加权平均池化)

使用 e~T~, e~S~, e~N~ 来计算**Triplet Loss**

### 和STARK比较

动态模版更新:

* STARK使用额外的分数预测头来决定是否更新
* FEAR使用无参数相似度模块来定义更新规则

双模版连接:

* STARK连接特征, 增加计算量
* FEAR使用一个可学习参数插值, 不增加计算



### 像素级融合模块

![image-20231009193307704](/Users/lizhi/Library/Application Support/typora-user-images/image-20231009193307704.png)

1. 将 search image feature 过一个 3*3 Conv-BN-ReLU 模块,
2. 然后跟 Template feature 进行点对点交叉相关 (Point-wise cross-correlation) 计算,
3. 将计算的到的特征图与 Search image feature 进行拼接
4. 过一个 1*1 Conv-BN-ReLU 来聚合



### 分类和边界框回归头

![image-20231010161959015](/Users/lizhi/Library/Application Support/typora-user-images/image-20231010161959015.png)

* bounding box regression network: 2 个 3*3 Conv-BN-ReLU 
* classification head: 2 个 3*3 Conv-BN-ReLU 



## Loss Function

### triplet loss

* 确保模型在特征空间中将同一类别的样本聚集在一起, 同时将不同类别的样本分开
* 鼓励模型使同一类别的样本距离更近, 不同类别的样本距离更远

$$
L_t = max\{d(e^T, e^S) - d(e^T, e^N) + margin, 0\}
$$

* 欧几里得距离 (L2距离)

$$
d(x_i, y_i) = \|x - y\|_2
$$

### regression loss

* 衡量模型预测的边界框与实际目标边界框之间的差异
* 回归损失是 1 减去交并比的值, 表明模型希望预测的边界框与实践目标边界框之间的IoU尽可能接近1

$$
L_{reg} = 1 - \sum_{i} IoU(t_{reg}, p_{reg})
$$

### Focal Loss (分类损失)

1. 确保模型正确分类图像的能力
2. y表示图像的真实类别, p表示模型预测该类别的概率
3. Focal Loss 专注于难以分类的样本, 通过 γ 来调整难易样本的重要性

$$
L_c = -(1-p_t)^γlog(p_t), \quad p_t = \begin{cases}
1 & \text{if } y = 1 \\
1 - p & \text{otherwise}
\end{cases}
$$

### overall loss

* is a linear combination of the three componets

* λ1 = 0.5,  λ2 = 1,  λ3 = 1

$$
L = λ_1 * L_t + λ_2 * L_{reg} + λ_3 * L_c
$$







