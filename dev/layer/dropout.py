import torch.nn as nn
import torch

class YOLODropout(nn.Module):
    """
    YOLO-Uyumlu Dropout implementasyonu.
    
    Bu sınıf, özellikle YOLO'nun DualDDetect yapısıyla çalışacak şekilde 
    tasarlanmıştır. Model içinde feature map'ler dolaşırken branching
    yapısını bozmaz ve tensor boyutlarını değiştirmez.
    """
    def __init__(self, p=0.4):
        """
        Initialize the specialized YOLO Dropout layer.
        
        Args:
            p (float): Probability of an element to be zeroed. Default: 0.4
        """
        super().__init__()
        self.p = p  # Save dropout probability
        self.np = 0  # No params to count
        
    def forward(self, x):
        """
        Forward pass of YOLO-compatible dropout.
        
        Bu metot girdi olarak ne gelirse onu aynen döndürür (pass-through),
        ancak eğitim modundaysa ve p>0 ise eğitim aşamasında dropout uygular.
        
        Args:
            x: Herhangi bir tipte input (tensor, list, tuple)
            
        Returns:
            Aynı yapıda ve boyutta output
        """
        # Eğer eğitimde değilse veya p=0 ise, dokunmadan çık
        if not self.training or self.p <= 0:
            return x
            
        # Girdinin tipine göre dropout uygula
        if isinstance(x, (list, tuple)):
            return type(x)(self._apply_dropout(item) for item in x)
        else:
            return self._apply_dropout(x)
    
    def _apply_dropout(self, x):
        """
        Tek bir tensor'a dropout uygula.
        
        Bu özel implementasyon, orijinal PyTorch dropout'tan farklı olarak:
        - Düzgün çalıştığından emin olmak için nn.functional.dropout kullanır
        - Dropout mask'ini manuel olarak oluşturur (channel-wise dropout için)
        - Rescale uygulanarak toplam aktivasyon değerini korur
        """
        if not torch.is_tensor(x):
            return x
            
        device = x.device
        # Channel-wise dropout maski oluştur (batch ve channel eksenlerinde)
        if len(x.shape) == 4:  # [B, C, H, W]
            mask = torch.rand(x.shape[0], x.shape[1], 1, 1, device=device) > self.p
            mask = mask.float() / (1 - self.p)  # Rescaling
            return x * mask  # Broadcasting ile tüm H,W'ye uygulanır
        else:
            # Normal dropout (diğer tensor şekilleri için)
            return torch.nn.functional.dropout(x, self.p, training=True)

# Geriye doğru uyumluluk için orijinal adı da kullan
Dropout = YOLODropout