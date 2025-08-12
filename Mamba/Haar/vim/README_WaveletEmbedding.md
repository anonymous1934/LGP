# WaveletEmbedding 模块化使用指南

## 概述

`WaveletEmbedding` 是一个模块化的小波嵌入模块，可以轻松集成到各种 Mamba 变种中，为模型提供全局-局部特征融合能力。

## 主要特性

- 🔧 **即插即用**: 可以轻松集成到任何 Mamba 变种中
- 🎛️ **高度可配置**: 支持不同小波函数、分解层数、融合策略
- 🎯 **灵活使用**: 可选择仅使用全局特征或进行特征融合
- 🔄 **向后兼容**: 不影响原有模型的使用方式
- 🚀 **性能优化**: 模块化设计便于性能调优

## 快速开始

### 1. 基本用法

```python
from models_mamba import WaveletEmbedding

# 创建小波嵌入模块
wavelet_embed = WaveletEmbedding(
    embed_dim=192,          # 嵌入维度
    img_size=224,           # 输入图像尺寸
    patch_size=16,          # patch大小
    wavelet_name="haar",    # 小波函数名称
    wavelet_levels=4,       # 小波分解层数
    use_fusion=True         # 是否使用自适应融合
)

# 使用示例
x = torch.randn(2, 3, 224, 224)  # 原始图像
global_features = wavelet_embed.get_global_features_only(x)
```

### 2. 在VisionMamba中使用

```python
from models_mamba import VisionMamba

# 创建带小波嵌入的模型
model = VisionMamba(
    embed_dim=192,
    depth=24,
    use_wavelet_embedding=True,    # 启用小波嵌入
    wavelet_name="haar",           # 小波类型
    wavelet_levels=4,              # 分解层数
    wavelet_fusion=True            # 启用融合
)

# 创建传统模型（不使用小波嵌入）
traditional_model = VisionMamba(
    embed_dim=192,
    depth=24,
    use_wavelet_embedding=False    # 禁用小波嵌入
)
```

### 3. 集成到自定义Mamba变种

```python
import torch.nn as nn
from models_mamba import WaveletEmbedding

class CustomMamba(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        
        # 基本组件
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        
        # 集成小波嵌入模块
        self.wavelet_embedding = WaveletEmbedding(
            embed_dim=embed_dim,
            img_size=224,
            patch_size=16,
            wavelet_name="db4",        # 可以使用不同的小波
            wavelet_levels=3,          # 自定义层数
            use_fusion=True
        )
        
        # 其他组件...
        
    def forward(self, x):
        original_x = x  # 保存原始输入
        
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # 使用小波嵌入增强特征
        x = self.wavelet_embedding(original_x, x)
        
        # 后续处理...
        return x
```

## 参数说明

### WaveletEmbedding 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `embed_dim` | int | 192 | 嵌入维度 |
| `img_size` | int | 224 | 输入图像尺寸 |
| `patch_size` | int | 16 | patch大小 |
| `wavelet_name` | str | "haar" | 小波函数名称 |
| `wavelet_levels` | int | 4 | 小波分解层数 |
| `use_fusion` | bool | True | 是否使用自适应融合 |
| `device` | torch.device | None | 计算设备 |

### VisionMamba 新增参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_wavelet_embedding` | bool | True | 是否使用小波嵌入 |
| `wavelet_name` | str | "haar" | 小波函数名称 |
| `wavelet_levels` | int | 4 | 小波分解层数 |
| `wavelet_fusion` | bool | True | 是否使用融合网络 |

## 支持的小波函数

- `"haar"`: Haar小波
- `"db1"`, `"db4"`, `"db8"`: Daubechies小波
- `"bior2.2"`, `"bior4.4"`: 双正交小波
- `"coif2"`, `"coif4"`: Coiflets小波
- 更多小波函数请参考 PyWavelets 文档

## 方法说明

### WaveletEmbedding 方法

#### `forward(x, patch_embeddings=None)`
主要的前向传播方法
- `x`: 原始图像张量 [B, C, H, W]
- `patch_embeddings`: 可选的patch嵌入 [B, N, D]
- 返回: 融合后的特征或全局特征

#### `get_global_features_only(x)`
仅获取全局小波特征
- `x`: 原始图像张量 [B, C, H, W]
- 返回: 全局小波特征 [B, embed_dim, H', W']

## 性能考虑

1. **内存使用**: 小波分解会增加内存使用，建议根据GPU内存调整 `wavelet_levels`
2. **计算复杂度**: 更多的分解层数会增加计算时间
3. **批处理大小**: 建议根据小波嵌入的内存需求调整批处理大小

## 最佳实践

1. **选择合适的小波函数**: 
   - `"haar"`: 简单快速，适合初步实验
   - `"db4"`: 平衡性能和质量
   - `"bior2.2"`: 适合保持细节信息

2. **调整分解层数**:
   - 较少层数 (2-3): 计算快，内存少
   - 较多层数 (4-5): 更丰富的多尺度信息

3. **融合策略**:
   - 设置 `use_fusion=True` 获得更好的性能
   - 设置 `use_fusion=False` 仅使用全局特征

## 注意事项

1. 确保安装了 `pytorch_wavelets` 库
2. 小波变换需要CUDA支持以获得最佳性能
3. 输入图像尺寸应该能被patch_size整除
4. 分解层数不宜过多，以免造成过度的内存消耗

## 示例代码

查看 `example_usage.py` 文件获取完整的使用示例。 