# Joint Feature Learning and Relation Modeling for Tracking One-Stream Framework (OSTrack)

（2022年发表在ECCV）

## 动机

目前流行的跟踪框架：两流两阶段跟踪框架（分别==提取模板==和==搜索区域==特征，然后进行关系建模）

​	提取的特征缺乏对目标的感知，目标-背景辨别能力有限

### 提出一种新的单流（OSTrack）框架

将特征学习和关系建模统一起来



## 贡献

1. 提出了一个简洁高效的但流的跟踪框架
2. 在Transformer的多头注意力机制中加入了，Early Candidate Elimination模块，从而加快了模型的推理速度
3. 在ji