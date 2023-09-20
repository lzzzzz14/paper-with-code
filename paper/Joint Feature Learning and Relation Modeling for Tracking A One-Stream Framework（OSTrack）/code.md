* 输入template：z，search：x
* z（B, 3, 224, 224），x（B, 3, 224, 224）

```python
x = self.patch_embed(x)
z = self.patch_embed(z)
```

```python
img_size = 224, 
patch_size = 16,
in_chans = 3,
embed_dim = 768,
self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
```

```python
embed_layer = PatchEmbed()
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

```python
x = self.proj(x)	# x (B, 768, 14, 14)
if self.flatten:
    x = x.flatten(2).transpose(1, 2)  # x (B, 768, 196) -> (B, 196, 768)
x = self.norm(x)
```

* 处理注意力掩码（attention mask）

```python
'''
首先，通过F.interpolate函数对mask_z和mask_x进行插值操作，将其尺寸缩放到与补丁化特征相匹配。这是因为在ViT模型中，注意力掩码需要与补丁化特征的尺寸相对应。

然后，将插值后的mask_z和mask_x转换为布尔类型，并将其展平为二维张量。这样做是为了与补丁化特征的形状相匹配，以便在自注意力机制中进行逐元素的注意力计算。

接下来，通过combine_tokens函数将mask_z和mask_x进行组合。这个函数的作用是将两个注意力掩码按照一定的方式进行组合，以得到最终的注意力掩码。mode参数指定了组合方式，具体的实现细节可能需要查看combine_tokens函数的定义。

最后，通过squeeze操作将最终的注意力掩码从三维张量中去除最后一个维度，以便与后续的计算兼容。
'''
if mask_z is not None and mask_x is not None:
	mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
    mask_z = mask_z.flatten(1).unsqueeze(-1)
        
    mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
    mask_x = mask_x.flatten(1).unsqueeze(-1)
    
    mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
    mask_x = mask_x.squeeze(-1)
```

* 添加分类令牌（classification token），以区分输入的x和z

```python
'''
首先，通过调用self.cls_token.expand(B, -1, -1)，将分类令牌self.cls_token在第一个维度（样本维度）上进行扩展，使其形状与特征张量x和z的形状保持一致。这样做是为了将分类令牌应用到每个样本中。

然后，通过cls_tokens + self.cls_pos_embed将分类令牌与位置嵌入（position embedding）相加。位置嵌入是一个可学习的参数，用于表示分类令牌的位置信息。通过将位置嵌入与分类令牌相加，可以将位置信息融入到分类令牌中。
'''
if self.add_cls_token:
    cls_tokens = self.cls_token.expand(B, -1, -1)
    cls_tokens = cls_tokens + self.cls_pos_embed
```

* 为z和x添加位置嵌入（position embedding）

```python
'''
首先，通过self.pos_embed_z和self.pos_embed_x获取位置嵌入张量。这些位置嵌入张量的形状为[1, num_patches + num_tokens, embed_dim]，其中num_patches是补丁化特征的数量，num_tokens是分类令牌的数量，embed_dim是嵌入维度。

然后，通过z += self.pos_embed_z和x += self.pos_embed_x将位置嵌入张量加到输入的z和x上。这样做是为了将位置信息融入到输入特征中，以帮助模型学习特征之间的空间关系。
'''
z += self.pos_embed_z
x += self.pos_embed_x
```

* 为z和x添加分割位置嵌入（segment position embedding）

```python
'''
如果模型的add_sep_seg属性为True，则会执行以下操作：

首先，通过self.search_segment_pos_embed获取搜索特征的分割位置嵌入张量。这个位置嵌入张量的形状为[1, num_patches + num_tokens, embed_dim]，其中num_patches是补丁化特征的数量，num_tokens是分类令牌的数量，embed_dim是嵌入维度。

然后，通过self.template_segment_pos_embed获取模板特征的分割位置嵌入张量，形状与搜索特征的分割位置嵌入张量相同。

最后，将搜索特征的分割位置嵌入张量加到输入的特征张量x上，将模板特征的分割位置嵌入张量加到输入的特征张量z上。这样做是为了将分割位置信息融入到特征张量中，以帮助模型学习特征之间的分割关系。
'''
if self.add_sep_seg:
    x += self.search_segment_pos_embed
    z += self.template_segment_pos_embed
```

* 合并z和x

```python
x = combine_token(z, x, mode=self.cat_mode)	# self.cat_mode = 'direct'
```

```python
'''
将z和x沿着dim=1进行拼接，x -> （B，392，768）
'''
if mode == 'direct':
	merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
```

* dropout

```python
x = self.pos_drop(x)
```

* 获取位置嵌入（position embedding）张量的长度

```python
'''
在这个代码片段中，self.pos_embed_z和self.pos_embed_x分别是搜索特征和模板特征的位置嵌入张量。这些位置嵌入张量的形状为[1, num_patches + num_tokens, embed_dim]，其中num_patches是补丁化特征的数量，num_tokens是分类令牌的数量，embed_dim是嵌入维度。

通过self.pos_embed_z.shape[1]和self.pos_embed_x.shape[1]，可以获取搜索特征和模板特征的位置嵌入张量的长度。这个长度表示了位置嵌入张量中的位置数量，即补丁化特征和分类令牌的总数。
'''
lens_z = self.pos_embed_z.shape[1]
lens_x = self.pos_embed_x.shape[1]
```

* 创建一个全局索引张量global_index_t，用于在模型的每个块中跟踪搜索特征的位置

```python
'''
首先，通过torch.linspace(0, lens_z - 1, lens_z)创建一个等差数列，从0到lens_z - 1，共lens_z个元素。这个等差数列表示搜索特征的位置索引。

然后，通过.to(x.device)将这个等差数列张量移动到与输入特征x相同的设备上。

最后，通过.repeat(B, 1)将这个等差数列张量在第一个维度（样本维度）上进行复制，复制B次，以适应输入特征x的批次大小。
'''
global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
global_index_t = global_index_t.repeat(B, 1)
```

* 创建一个全局索引张量global_index_s，并将其重复B次，以跟踪模板特征的位置

```python
'''
首先，通过torch.linspace(0, lens_x - 1, lens_x)创建一个等差数列，从0到lens_x - 1，共lens_x个元素。这个等差数列表示模板特征的位置索引。

然后，通过.to(x.device)将这个等差数列张量移动到与输入特征x相同的设备上。

接下来，通过.repeat(B, 1)将这个等差数列张量在第一个维度（样本维度）上进行复制，复制B次，以适应输入特征x的批次大小。

最后，创建一个空列表removed_indexes_s，用于存储后续迭代中每个块中的移除的索引。

与之前的global_index_t不同，global_index_s是用于跟踪模板特征的位置的全局索引张量。
'''
global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
global_index_s = global_index_s.repeat(B, 1)
removed_indexes_s = []
```

