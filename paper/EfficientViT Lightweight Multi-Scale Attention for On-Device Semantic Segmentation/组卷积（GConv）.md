# group convolution (分组卷积)

**普通卷积**

![img](https://upload-images.jianshu.io/upload_images/17002688-638cec40510d9bef.png?imageMogr2/auto-orient/strip|imageView2/2/w/459/format/webp)

为方便理解，图中只有一个卷积核，此时输入输出数据为：
 **输入feature map尺寸**： W×H×C  ，分别对应feature map的宽，高，通道数；
 **单个卷积核尺寸**： k×k×C ，分别对应单个卷积核的宽，高，通道数；
 **输出feature map尺寸** ：W'×H' ，输出通道数等于卷积核数量，输出的宽和高与卷积步长有关，这里不关心这两个值。

参数量 ![params=k^2C](https://math.jianshu.com/math?formula=params%3Dk%5E2C)
运算量![FLOPs=k^2CW'H'](https://math.jianshu.com/math?formula=FLOPs%3Dk%5E2CW%27H%27)，这里只考虑浮点乘数量，不考虑浮点加。

**分组卷积**

![img](https://upload-images.jianshu.io/upload_images/17002688-3c6c6ad17fc9b5d9.png?imageMogr2/auto-orient/strip|imageView2/2/w/474/format/webp)

将图一卷积的输入feature map分成![g](https://math.jianshu.com/math?formula=g)组，每个卷积核也相应地分成![g](https://math.jianshu.com/math?formula=g)组，在对应的组内做卷积，如上图2所示，图中分组数![g=2](https://math.jianshu.com/math?formula=g%3D2)，即上面的一组feature map只和上面的一组卷积核做卷积，下面的一组feature map只和下面的一组卷积核做卷积。每组卷积都生成一个feature map，共生成![g](https://math.jianshu.com/math?formula=g)个feature map。

**输入每组feature map尺寸：**![W×H× \frac {C} {g}](https://math.jianshu.com/math?formula=W%C3%97H%C3%97%20%5Cfrac%20%7BC%7D%20%7Bg%7D) ，共有![g](https://math.jianshu.com/math?formula=g)组；
 **单个卷积核每组的尺寸：**![k×k×\frac {C} {g}](https://math.jianshu.com/math?formula=k%C3%97k%C3%97%5Cfrac%20%7BC%7D%20%7Bg%7D)，一个卷积核被分成了![g](https://math.jianshu.com/math?formula=g)组；
 **输出feature map尺寸：**![W'×H'×g](https://math.jianshu.com/math?formula=W'%C3%97H'%C3%97g)，共生成![g](https://math.jianshu.com/math?formula=g)个feature map。

现在我们再来计算一下分组卷积时的参数量和运算量：
 参数量 ![params=k^2×\frac {C} {g}×g=k^2C](https://math.jianshu.com/math?formula=params%3Dk%5E2%C3%97%5Cfrac%20%7BC%7D%20%7Bg%7D%C3%97g%3Dk%5E2C)
 运算量![FLOPs=k^2×\frac {C} {g}×W'×H'×g=k^2CW'H'](https://math.jianshu.com/math?formula=FLOPs%3Dk%5E2%C3%97%5Cfrac%20%7BC%7D%20%7Bg%7D%C3%97W'%C3%97H'%C3%97g%3Dk%5E2CW'H')

### 用同等的参数量生成了g个feature map