import torch.nn as nn
import torch
import torch.nn.functional as F
"""
def squeeze_excite_block(input, ratio=16):
    filters = input.shape[-1]  # Çıktı kanal sayısı
    se = GlobalAveragePooling2D()(input)  # Global average pooling
    se = Dense(filters // ratio, activation='relu')(se)  # Sıkıştırma (Dense katman)
    se = Dense(filters, activation='sigmoid')(se)  # Yeniden genişletme (Dense katman)
    return Multiply()([input, se])  # Kanal başına ağırlıklandırma
"""
class SE(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(SE, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        #bias True olmayınca loss nan oluyor, pooling ile beraber True olması tavsiye ediliyormuş
        self.fc1 = nn.Linear(in_channels, in_channels // ratio, bias=True) # activation'sız dense ile linear aynıymış, devamında activation ekleniyor
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // ratio, in_channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_shape = len(x.size())
        
        if input_shape == 2:  # [B, C] (already pooled)
            b, c = x.size()
            y = x  # Already pooled, so use directly
        else:  # [B, C, H, W]
            b, c, _, _ = x.size()
            y = self.global_avg_pool(x)
            y = y.view(b, c)  # [B, C, 1, 1] -> [B, C]
        
        # Rest of the processing
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Apply channel attention
        if input_shape == 2:  # [B, C]
            return x * y  # Element-wise multiplication
        else:  # [B, C, H, W]
            y = y.view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
            return x * y
# Attention
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, bias=True)

    def forward(self, x):
        input_shape = len(x.size())
        
        if input_shape == 2:  # [B, C] (already pooled)
            B, C = x.shape
            # Reshape to sequence length 1 for attention
            x = x.unsqueeze(1)  # [B, C] -> [B, 1, C]
            x = x.permute(1, 0, 2)  # [B, 1, C] -> [1, B, C]
            attn_output, _ = self.attn(x, x, x)  # [1, B, C]
            attn_output = attn_output.permute(1, 0, 2)  # [1, B, C] -> [B, 1, C]
            return attn_output.squeeze(1)  # [B, 1, C] -> [B, C]
        else:  # [B, C, H, W]
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, H*W, C]
            x = x.permute(1, 0, 2)  # [B, H*W, C] -> [H*W, B, C]
            attn_output, _ = self.attn(x, x, x)  # [H*W, B, C]
            attn_output = attn_output.permute(1, 0, 2)  # [H*W, B, C] -> [B, H*W, C]
            attn_output = attn_output.permute(0, 2, 1)  # [B, H*W, C] -> [B, C, H*W]
            return attn_output.view(B, C, H, W)  # [B, C, H*W] -> [B, C, H, W]

# layernorm'da sonuncunun channel sayısı olması gerekiyormuş, bunda ona uygun şekilde dönüştürüyoruz
# custom olanın forward'ta ayarlanması gereken değerlerine yeniden bakılacak
class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        input_shape = len(x.size())
        
        if input_shape == 2:  # [B, C]
            return self.ln(x)  # Apply directly
        else:  # [B, C, H, W]
            x = x.permute(0, 2, 3, 1)  # -> [B, H, W, C]
            x = self.ln(x)
            x = x.permute(0, 3, 1, 2)  # -> [B, C, H, W]
            return x