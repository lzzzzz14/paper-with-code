# 深度可分离卷积（Depthwise separable convolution）

* 参数数量和运算成本比较低

**传统卷积**

![img](https://pic3.zhimg.com/80/v2-57175d446973a83f7a58fc9c21af12e2_720w.webp)

卷积层共4个Filter，每个Filter包含3个Kernel，每个Kernel大小为3×3。

卷积层**参数数量**可用如下公式来计算（卷积核W × 卷积核H × 输入通道数 × 输出通道数）：

N_std = 4 × 3 × 3 × 3 = 108

**计算量**（卷积核W × 卷积核H × （图片W - 卷积核W +1） × （图片H - 卷积核H + 1） × 输入通道数 × 输出通道数）：

C_std = 3\*3\*(5-2)\*(5-2)\*3\*4 = 972

### 深度可分离卷积

可分为两个过程：逐通道卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）

**逐通道卷积**

Depthwise Convolution的一个卷积核负责一个通道，一个通道只被一个卷积核卷积，这个过程产生的feature map通道数和输入的通道数完全一样

一张5×5像素、三通道彩色输入图片（shape为5×5×3），Depthwise Convolution首先经过第一次卷积运算，DW完全是在二维平面内进行。卷积核的数量与上一层的通道数相同（通道和卷积核一一对应）。所以一个三通道的图像经过运算后生成了3个Feature map(如果有same padding则尺寸与输入层相同为5×5)，如下图所示。（卷积核的shape即为：**卷积核W x 卷积核H x 输入通道数**）

![img](https://pic2.zhimg.com/80/v2-2bdf9cb05d9caf6c968c43610f6b8b95_720w.webp)

其中一个Filter只包含一个大小为3×3的Kernel，卷积部分的参数个数计算如下（即为：**卷积核Wx卷积核Hx输入通道数**）：

N_depthwise = 3 × 3 × 3 = 27

计算量为（即：**卷积核W x 卷积核H x (图片W-卷积核W+1) x (图片H-卷积核H+1) x 输入通道数**）

C_depthwise=3x3x(5-2)x(5-2)x3=243

Depthwise Convolution完成后的Feature map数量与输入层的通道数相同，无法扩展Feature map。而且这种运算对输入层的每个通道独立进行卷积运算，没有有效的利用不同通道在相同空间位置上的feature信息。因此需要Pointwise Convolution来将这些Feature map进行组合生成新的Feature map

**逐点卷积**

Pointwise Convolution的运算与常规卷积运算非常相似，它的卷积核的尺寸为 1×1×M，M为上一层的通道数。所以这里的卷积运算会将上一步的map在深度方向上进行加权组合，生成新的Feature map。有几个卷积核就有几个输出Feature map。（卷积核的shape即为：**1 x 1 x 输入通道数 x 输出通道数**）

![img](https://pic4.zhimg.com/80/v2-7593e8b0c43db44d62f19fec7c8795bb_720w.webp)

由于采用的是1×1卷积的方式，此步中卷积涉及到的参数个数可以计算为(即为：**1 x 1 x 输入通道数 x 输出通道数**）：

N_pointwise = 1 × 1 × 3 × 4 = 12

计算量(即为：**1 x 1 特征层W x 特征层H x 输入通道数 x 输出通道数**）：

C_pointwise = 1 × 1 × 3 × 3 × 3 × 4 = 108

经过Pointwise Convolution之后，同样输出了4张Feature map，与常规卷积的输出维度相同。

### 参数对比

常规卷积的参数个数为：

N_std = 4 × 3 × 3 × 3 = 108



Separable Convolution的参数由两部分相加得到：

N_depthwise = 3 × 3 × 3 = 27

N_pointwise = 1 × 1 × 3 × 4 = 12

N_separable = N_depthwise + N_pointwise = 39



相同的输入，同样是得到4张Feature map，Separable Convolution的参数个数是常规卷积的约1/3。因此，在参数量相同的前提下，采用Separable Convolution的神经网络层数可以做的更深。

### 计算量对比

常规卷积的计算量为：

C_std =3*3*(5-2)*(5-2)*3*4=972



Separable Convolution的参数由两部分相加得到：

C_depthwise=3x3x(5-2)x(5-2)x3=243

C_pointwise = 1 × 1 × 3 × 3 × 3 × 4 = 108

C_separable = C_depthwise + C_pointwise = 351



相同的输入，同样是得到4张Feature map，Separable Convolution的计算量是常规卷积的约1/3。因此，在计算量相同的情况下，Depthwise Separable Convolution可以将神经网络层数可以做的更深。