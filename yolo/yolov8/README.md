# yolov8代码

### 简单的使用

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format	
```

接下来看一下细节,

* 在ultralytics/models/yolo/model.py中

```python
class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': ClassificationModel,
                'trainer': yolo.classify.ClassificationTrainer,
                'validator': yolo.classify.ClassificationValidator,
                'predictor': yolo.classify.ClassificationPredictor, },
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, }, }
```

这次我们学的是detect模型

* 首先查看 'model' DetecgtionModel

* 在ultralytics/nn/tasks.py中

**class DetectionModel (BaseModel)**

我们传入的cfg: “yolov8n.yaml”是一个dict

```python
# Define model
self.yaml = cfg 	# 将cfg赋值给self.yaml
ch = sekf.yaml['ch']	# 将配置文件中的通道数拿出来, 作为input channels
self.yaml['nc'] = nc # number of classes

# deepcopy创建配置文件的副本
# parse_model会根据配置文件中的信息创建模型, 返回一个元组, 第一个元素是构建的模型对象, 第二个元素是保存模型的列表
# self.model是构建的模型对象, self.saveshi一个保存模型的列表
self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)

# self.names是一个字典, 用于存储类别的名字
self.namse = {i: f'{i}' for i in range(self.yaml['nc'])}
```

* 看一下是如何根据配置文件构建模型的

**def parse_model**

```python
# 输入参数 d:model_dict, ch:input_channels(3), verbose=True(是否打印详细参数)

# d.get(x)从字典d中获取键为x的值, 分别对应'nc', 'activation', 'sacles' ...
# nc:类别数, act:表示激活函数, scales:模型的尺度
nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))

# 检查配置文件中是否存在sacles这个键(注意和前面的scales进行区分), 如果存在, 则复制给scale
if scales:
    scale = d.get('scale')
    if not scale:
      # 从scalse字典中获取第一个键, 如果为空则应发下面的警告, 并假设scale的值为第一个键的值
      scale = tuple(scales.keys())[0]
      LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
  # 从scales字典中根据scale的值获取对应的三个值
  depth, width, max_channels = scales[scale]

# 选择激活函数
if act:
  	# eval()函数将act的值作为字符串进行求值
    Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
    if verbose:
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print

# 初始化一些值
ch = [ch]
layers, save, c2 = [], [], ch[-1]
```

* 根据参数来构建所有的模块

```python
for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
    m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
    for j, a in enumerate(args):
        if isinstance(a, str):
            with contextlib.suppress(ValueError):
                args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

    n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
    
    # 来变化c1, c2的值
    if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
             BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
        c1, c2 = ch[f], args[0]	# 对c1, c2进行赋值
        if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
            c2 = make_divisible(min(c2, max_channels) * width, 8)

        args = [c1, c2, *args[1:]]
        if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
            args.insert(2, n)  # number of repeats
            n = 1
    elif m is AIFI:
        args = [ch[f], *args]
    elif m in (HGStem, HGBlock):
        c1, cm, c2 = ch[f], args[0], args[1]
        args = [c1, cm, c2, *args[2:]]
        if m is HGBlock:
            args.insert(4, n)  # number of repeats
            n = 1

    elif m is nn.BatchNorm2d:
        args = [ch[f]]
    elif m is Concat:
        c2 = sum(ch[x] for x in f)
    elif m in (Detect, Segment, Pose):
        args.append([ch[x] for x in f])
        if m is Segment:
            args[2] = make_divisible(min(args[2], max_channels) * width, 8)
    elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
        args.insert(1, [ch[x] for x in f])
    else:
        c2 = ch[f]
    
   	# 根据m的值来构建模块, 根据n的值来构建对应数量的模块
    m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
    
    # 获取模块对象的类型名称
    t = str(m)[8:-2].replace('__main__.', '')  # module type
    
    # 模块对象 m_ 被扩展了三个属性：np、i 和 f
    # 用于记录模块对象的参数数量、索引和类型
    m.np = sum(x.numel() for x in m_.parameters())  # number params
    m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
    if verbose:
        LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
    save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
    layers.append(m_)	# 将m变为的module添加到layer这个列表中
    if i == 0:
        ch = []
    ch.append(c2)	# 将上一层的out_c添加到ch中, 作为下一层的in_c
```

* 最后输出

```python
return nn.Sequential(*layers), sorted(save)
```