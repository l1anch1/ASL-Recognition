"""Vision Transformer模型实现"""

import torch
import torch.nn as nn
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.config import VIT_NUM_HEADS, VIT_PATCH_SIZE, VIT_EMBED_DIM, VIT_DEPTH, NUM_CLASSES

class PatchEmbedding(nn.Module):
    """将图像分割成patches并进行线性嵌入"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """x: (B, C, H, W)"""
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 用于存储注意力权重以便可视化
        self.attention_weights = None

    def forward(self, x):
        """x: (B, N, embed_dim)"""
        B, N, C = x.shape

        # 生成query, key, value
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, head_dim)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        
        # 保存注意力权重以便可视化
        self.attention_weights = attn.detach()
        
        attn = self.attn_dropout(attn)

        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """多层感知器模块"""

    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTNet(nn.Module):
    """用于ASL手语识别的Vision Transformer模型"""

    def __init__(
        self,
        img_size=32,
        patch_size=VIT_PATCH_SIZE,
        in_channels=3,
        num_classes=NUM_CLASSES,
        embed_dim=VIT_EMBED_DIM,
        depth=VIT_DEPTH,
        num_heads=VIT_NUM_HEADS,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        # 图像处理为patches
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.n_patches
        self.patch_size = patch_size
        self.img_size = img_size

        # 添加可学习的分类token ([CLS])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer编码器
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # 规范化层
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 应用到所有模块
        self.apply(self._init_weights_recursive)

    def _init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # (B, C, H, W) -> (B, n_patches, embed_dim)
        x = self.patch_embed(x)
        B, n, _ = x.shape

        # 添加分类token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (B, 1 + n_patches, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过Transformer编码器
        for block in self.blocks:
            x = block(x)

        # 应用规范化
        x = self.norm(x)

        # 使用[CLS]令牌进行分类
        x = x[:, 0]  # (B, embed_dim)

        # 分类头
        x = self.head(x)  # (B, num_classes)
        return x
    
    def get_attention_weights(self):
        """获取最后一个transformer块的注意力权重"""
        return self.blocks[-1].attn.attention_weights
