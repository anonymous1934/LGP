"""
WaveletEmbedding 模块化使用示例
"""

import torch
import torch.nn as nn

# 示例：在其他Mamba变种中使用WaveletEmbedding
class CustomMambaWithWavelet(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 导入WaveletEmbedding
        from models_mamba import WaveletEmbedding
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        
        # 小波嵌入模块 - 即插即用
        self.wavelet_embedding = WaveletEmbedding(
            embed_dim=embed_dim,
            img_size=224,
            patch_size=16,
            wavelet_name="haar",
            wavelet_levels=4,
            use_fusion=True
        )
        
        # 其他组件
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1000)
        
    def forward(self, x):
        original_x = x
        
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # 添加cls token
        cls_token = torch.zeros(x.shape[0], 1, self.embed_dim, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        
        # 使用小波嵌入进行特征增强
        x = self.wavelet_embedding(original_x, x)
        
        # 后续处理
        x = self.norm(x)
        return self.head(x[:, 0])

if __name__ == "__main__":
    # 测试模型
    model = CustomMambaWithWavelet()
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输出形状: {output.shape}") 