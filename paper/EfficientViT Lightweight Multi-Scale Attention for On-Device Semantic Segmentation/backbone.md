**efficient_backbone_b0**

```python
width_list=[8, 16, 32, 64, 128]，
depth_list=[1, 2, 2, 2, 2],
dim=16
```

**class EfficientViTBackbone**

## input stem

```python
# input stem
self.input_stem = [
    ConvLayer(
        in_channels=3,
        out_channels=width_list[0], # backbone0 : 8
        stride=2,
        norm=norm,
        act_func=act_func,
    )
]
```

```python
for _ in range(depth_list[0]):	# depth_list[0]=1
    block = self.build_local_block(
        in_channels=width_list[0],	# width_list[0]=8
        out_channels=width_list[0],	# width_list[0]=8
        stride=1,
        expand_ratio=1,	# 指定block用什么，这里使用的是DSConv，里面包括一个组卷积和一个1×1卷积
        norm=norm,
        act_func=act_func,
    )
    self.input_stem.append(ResidualBlock(block, IdentityLayer()))
```

**build_local_block**

```python
# 过一个DSConv块
if expand_ratio == 1:
    block = DSConv(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        use_bias=(True, False) if fewer_norm else False,
        norm=(None, norm) if fewer_norm else norm,
        act_func=(act_func, None),
    )
```

**class DSConv**

```python
# 这是一个组卷积
self.depth_conv = ConvLayer(
    in_channels,
    in_channels,
    kernel_size,
    stride,
    groups=in_channels,
    norm=norm[0],
    act_func=act_func[0],
    use_bias=use_bias[0],
)
```

```python
# 这是一个1*1卷积
self.point_conv = ConvLayer(
    in_channels,
    out_channels,
    1,	# kernel_size
    norm=norm[1],
    act_func=act_func[1],
    use_bias=use_bias[1],
)
```

```python
x = self.depth_conv(x)
x = self.point_conv(x)
```

**ResidualBlock**

```python
def forward_main(self, x: torch.Tensor) -> torch.Tensor:
    if self.pre_norm is None:	# pre_norm：预归一化
        return self.main(x)
    else:
        return self.main(self.pre_norm(x))

def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.main is None:
        res = x
    elif self.shortcut is None:	
        res = self.forward_main(x)
    else:	# 如果self.main和self.shortcut都有，则进行残差操作
        res = self.forward_main(x) + self.shortcut(x)
        if self.post_act:	# post_act：激活函数
            res = self.post_act(res)
    return res
```

## stage1 & stage2

```python
# stages
self.stages = []
for w, d in zip(width_list[1:3], depth_list[1:3]):
    # w = 16, 32，d = 2, 2
    stage = []
    for i in range(d):
        stride = 2 if i == 0 else 1
        block = self.build_local_block(
            in_channels=in_channels,
            out_channels=w,
            stride=stride,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func,
        )
        block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
        stage.append(block)
        in_channels = w
    self.stages.append(OpSequential(stage))
    self.width_list.append(in_channels)
```

