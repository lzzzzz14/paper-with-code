# ViT

[论文地址](arxiv.org/pdf/2010.11929.pdf)

[官方pytorch实现](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)

[ViT｜ Vision Transformer ｜理论 + 代码_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1xm4y1b7Pw/?spm_id_from=333.788&vd_source=78a547131858b1310aa0cefdfdab4b71)

[课件](https://65d8gk.axshare.com)

[代码](https://github.com/Enzo-MiMan/cv_related_collections/tree/main/classification/vision_transformer/vit_model.py)

![Figure 1 from paper](https://gitcode.net/mirrors/google-research/vision_transformer/-/raw/master/vit_figure.png)

## 过一遍ViT的前向传播

一些超参数

```python
model = VisionTran(img_size=224,	# 输入的图片是224*224
                   patch_size=16,	# patch是16*16
                   embed_dim=768,	# embed之后的通道数(16*16*3得出)
                   depth=12,			# 
                   num_heads=12, 	# 多头的数量
                   representation_size=768 if has_logits else None,
                   num_classes=num_classes)
```

1. patch embedding

```python
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)		# 图片的大小 -> 224*224*3
        patch_size = (patch_size, patch_size)	# patch的大小	-> 16*16*3
        self.img_size = img_size
        self.patch_size = patch_size
        # 一张图片被分为几个patch	-> 14*14
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])	
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 一个卷积层: input_channel=3, output_channel=768, kernel_size=16*16, stride=16
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 判断是否要过一个norm层 (默认是直接输出nn.Identity)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
	def forward(self, x):
        B, C, H, W = x.shape	# 先查看输入的形状 (8, 3, 224, 224), 判断输入的H*W是否匹配img_size[0]*img_size[1]
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]	
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
```

* 过一遍patchembed的前向传播
  1. 卷积: (8, 3, 224, 224) &rarr; (8, 768, 14, 14)						
  2. flatten[2]: (8, 768, 14, 14) &rarr; (8, 768, 196)
  3. Transpose(1, 2): (8, 768, 196) &rarr; (8, 196, 768) 

![image-20230823195756446](/Users/lizhi/Library/Application Support/typora-user-images/image-20230823195756446.png)

2. Class embedding + Positional Embedding

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))	# 通过nn.Parameter, 将clas_token作为一个可学习的参数	初始化: (1, 1, 768)
self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None	# 这个distilled默认是None, 所以dist_token默认也是None

# 这个是添加位置编码, 也是可学习的 num_patches=196, self.num_tokens=1, embed_dim=768
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
self.pos_drop = nn.Dropout(p=drop_ratio)	# 默认drop_ratio为0

# 这里开始, 上面的是一些定义
# [1, 1, 768] -> [B, 1, 768]
cls_token = self.cls_token.expand(x.shape[0], -1, -1)	# 添加一个分类的token, 后面也只输出这一个来进行分类, 通过expand将第一个维度转换为batch_size 
if self.dist_token is None:
  x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]	将cls_token加到x上, 在第二个维度, 也就是patch的数量 (cls_token排在第一个)
else:
	x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

# 将位置编码加入x中, 对应位置进行数值相加; 然后过dropout层(默认ratio为0)  
x = self.pos_drop(x + self.ppos_embed)	
```

* 过一遍形状变化
  1. x concat cls_token: (8, 196, 768) &rarr; (8, 197, 768)
  2. x + pos_drop: (8, 197, 768) 形状没有变化

![image-20230824101740193](/Users/lizhi/Library/Application Support/typora-user-images/image-20230824101740193.png)

3. Transformer Encoder

![image-20230824113728092](/Users/lizhi/Library/Application Support/typora-user-images/image-20230824113728092.png)

```python
x = self.blocks(x)		# 在vit中调用Block类(也就是transformer中的encoder块)

class Block(nn.Module):
	def forward(self, x):
    # x过一个norm层, 然后进入attn过多头自注意力, 然后是drop_path, 并且加上一个残差连接
    x = x + self.drop_path(self.attn(self.norm1(x)))
    # x过一个norm层, 然后过Mlp多层感知机, 之后还是一个drop_path, 加一个残差连接
    x = x + self.drop_path(self.mlp(self.norm2(x)))

class Attention(nn.Module):
  def forward(self, x):
    B, N, C = x.shape
    
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
```



































































