1. We use the first four stages of the neural network pretrained on the ImageNet [14] as

   a feature extraction module. 中的4阶段是什么意思

   在特征提取阶段, 过一个阶段就是分辨率减半, 过四个阶段就是分辨率降为原来的1/16

2. The output of the backbone network is a feature map of stride *16* for the template

   and search images. stride=16是什么意思

   对应前面的分辨率下降为原来的1/16, 这里用stride=16来指代

3. 