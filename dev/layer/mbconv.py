import torch
import torch.nn as nn

class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) - EfficientNet'in temel yapı taşı
    
    Args:
        c1 (int): Giriş kanalları
        c2 (int): Çıkış kanalları
        expand_ratio (float): Genişleme oranı, ara katmanın boyutunu belirler
        stride (int): Konvolüsyon adımı
        use_se (bool): Squeeze-Excitation bloğunu kullan (True) veya kullanma (False)
        se_ratio (float): Squeeze-Excitation blok oranı, ara katman boyutunu belirler
        dropout_rate (float): Droupout oranı (0-1 arası)
    """
    def __init__(self, c1, c2, expand_ratio=4.0, stride=1, use_se=True, se_ratio=0.25, dropout_rate=0.2):
        super(MBConv, self).__init__()
        
        # Genişleme oranını kontrol et
        self.use_residual = stride == 1 and c1 == c2
        expanded_c = int(c1 * expand_ratio)
        
        # Layers
        layers = []
        
        # Expansion (1x1 conv ile kanal sayısını artır)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(c1, expanded_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_c),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise Convolution (her kanal için ayrı 3x3 konvolüsyon)
        layers.extend([
            # Grupları expanded_c'ye eşit yaparak depthwise konvolüsyon elde ederiz
            nn.Conv2d(expanded_c, expanded_c, kernel_size=3, stride=stride, 
                      padding=1, groups=expanded_c, bias=False),
            nn.BatchNorm2d(expanded_c),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze and Excitation (opsiyonel)
        if use_se:
            se_c = max(1, int(c1 * se_ratio))
            layers.append(SEModule(expanded_c, se_c))
        
        # Projection (1x1 conv ile kanal sayısını düşür)
        layers.extend([
            nn.Conv2d(expanded_c, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2)
        ])
        
        # Dropout (opsiyonel)
        if self.use_residual and dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SEModule(nn.Module):
    """
    Squeeze-Excitation Module - Kanal boyutunda dikkat mekanizması
    
    Args:
        c (int): Giriş kanal sayısı
        c_reduced (int): Azaltılmış kanal sayısı (genellikle c/r şeklinde, r=8, 16 gibi)
    """
    def __init__(self, c, c_reduced):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c_reduced, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(c_reduced, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y


# Farklı MBConv varyasyonları - örnek kullanım için
class MBConvBlock(nn.Module):
    """
    MBConv'un sıralı blokları - Tekrarlı kullanım için
    
    Args:
        c1 (int): Giriş kanalları
        c2 (int): Çıkış kanalları
        n (int): Blok sayısı
        expand_ratio (float): Genişleme oranı
        stride (int): İlk bloğun adımı
    """
    def __init__(self, c1, c2, n=1, expand_ratio=4.0, stride=1):
        super(MBConvBlock, self).__init__()
        
        layers = []
        # İlk blok stride kullanabilir, sonrakiler her zaman stride=1 olacak
        layers.append(MBConv(c1, c2, expand_ratio, stride))
        
        # Kalan bloklar (hepsi stride=1 ve c1=c2)
        for _ in range(1, n):
            layers.append(MBConv(c2, c2, expand_ratio, 1))
            
        self.blocks = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.blocks(x)