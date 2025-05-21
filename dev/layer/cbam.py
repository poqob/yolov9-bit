import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Kanal Dikkat Modülü - Kanal boyutunda adaptif önem ağırlıkları üretir
    
    Args:
        channels (int): Giriş kanallarının sayısı
        reduction_ratio (int): Kanal sayısını azaltma oranı (genellikle 8 veya 16)
    """
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Paylaşılan MLP - kanal sayısını azaltıp sonra tekrar arttırır
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        # Ortalama havuzlama yoluyla kanal dikkat özellikleri
        avg_out = self.mlp(self.avg_pool(x))
        
        # Maksimum havuzlama yoluyla kanal dikkat özellikleri
        max_out = self.mlp(self.max_pool(x))
        
        # İki özellik haritasını birleştirip bir sigmoid fonksiyonu uygulayarak
        # her kanal için [0, 1] aralığında bir dikkat katsayısı üretir
        out = torch.sigmoid(avg_out + max_out)
        
        return out


class SpatialAttention(nn.Module):
    """
    Uzamsal Dikkat Modülü - Uzamsal (H,W) boyutlarda adaptif önem ağırlıkları üretir
    
    Args:
        kernel_size (int): Konvolüsyon katmanının çekirdek boyutu (genellikle 7)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        # Her konumda ortalama değer (kanal boyunca)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Her konumda maksimum değer (kanal boyunca)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Ortalama ve maksimum değerleri kanal boyutunda birleştir
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        
        # Konvolüsyon ve sigmoid ile [0, 1] aralığında uzamsal dikkat haritası üret
        spatial_attention = torch.sigmoid(self.conv(spatial_features))
        
        return spatial_attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Hem kanal hem de uzamsal dikkat mekanizmalarını sırayla uygular
    
    Args:
        channels (int): Giriş kanallarının sayısı
        reduction_ratio (int): Kanal dikkat modülündeki azaltma oranı
        spatial_kernel_size (int): Uzamsal dikkat modülündeki çekirdek boyutu
    """
    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        
    def forward(self, x):
        # Önce kanal dikkat modülünü uygula
        x = x * self.channel_attention(x)
        
        # Sonra uzamsal dikkat modülünü uygula
        x = x * self.spatial_attention(x)
        
        return x


# CBAM Entegre Edilmiş Residual Blok - Örnek kullanım
class CBAMResBlock(nn.Module):
    """
    CBAM entegre edilmiş ResNet tarzı residual blok
    
    Args:
        c1 (int): Giriş kanalları
        c2 (int): Çıkış kanalları
        shortcut (bool): Residual bağlantı kullanılsın mı?
    """
    def __init__(self, c1, c2, shortcut=True):
        super(CBAMResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        
        # CBAM dikkat modülü
        self.cbam = CBAM(c2)
        
        # Residual bağlantı için gerekirse boyut dönüşümü
        self.shortcut = shortcut
        if not shortcut or c1 != c2:
            self.downsample = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        # CBAM dikkat modülünü uygula
        out = self.cbam(out)
        
        # Residual bağlantı
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out, inplace=True)
        
        return out